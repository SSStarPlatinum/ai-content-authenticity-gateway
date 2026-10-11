# 模型集成指南

本文档说明如何将新的 AI 伪造检测模型集成到推理网关中。

## 1. 集成流程总览

```
获取模型权重 → 转换为 ONNX → 编写适配器 → 注册模型 → 配置启用 → 测试验证
```

## 2. 编写模型适配器

### 2.1 继承 BaseDetector

在 `app/models/` 下新建文件，例如 `my_detector.py`：

```python
from __future__ import annotations
import os
import numpy as np
from PIL import Image

from ..utils.image import resize_to_tensor
from .base import BaseDetector, DetectionResult
from .registry import register_model

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@register_model("my_detector")
class MyDetector(BaseDetector):
    """我的自定义检测模型。"""

    model_name = "My Custom Detector"
    version = "1.0.0"
    default_weight = 1.0
    strong_postprocessing = ["none", "jpeg"]  # 模型擅长的后处理类型

    WEIGHT_FILENAME = "my_detector.onnx"
    INPUT_SIZE = (224, 224)
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def _load(self) -> None:
        """加载 ONNX 权重。"""
        if ort is None:
            raise RuntimeError("onnxruntime not installed")

        weight_path = os.path.join(self.weights_dir, self.WEIGHT_FILENAME)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"weight not found: {weight_path}. "
                f"Place {self.WEIGHT_FILENAME} in {self.weights_dir}/"
            )

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            weight_path, sess_options=sess_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def _predict(self, img: Image.Image) -> DetectionResult:
        """单图推理。"""
        # 1. 预处理
        x = resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD)

        # 2. 推理
        outputs = self._session.run(None, {self._input_name: x})
        logits = outputs[0][0]  # 根据模型输出结构调整

        # 3. 后处理
        probs = self._softmax(logits)
        fake_prob = float(probs[1])  # 假设索引 1 是 fake 类
        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "real_prob": round(float(probs[0]), 6),
                "input_size": f"{self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]}",
            },
        )

    def batch_predict(self, images):
        """（可选）重写批量推理，提升吞吐。"""
        if not self._loaded:
            self.load()
        if not images:
            return []
        batch = np.concatenate(
            [resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD) for img in images],
            axis=0,
        )
        outputs = self._session.run(None, {self._input_name: batch})
        probs_batch = self._softmax(outputs[0], axis=1)
        return [
            DetectionResult(
                model_id=self.model_id,
                fake_prob=float(p[1]),
                label="fake" if p[1] > 0.5 else "real",
                confidence=float(abs(p[1] - 0.5) * 2),
            )
            for p in probs_batch
        ]
```

### 2.2 注册到自动发现

编辑 `app/models/registry.py`，在 `auto_discover()` 函数中添加模块路径：

```python
def auto_discover() -> None:
    modules = [
        "app.models.mock",
        "app.models.cnn_spotter",
        "app.models.universal_fake",
        "app.models.aeroblade",
        "app.models.freq_detector",
        "app.models.my_detector",  # ← 新增
    ]
```

### 2.3 关键接口说明

#### DetectionResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_id` | str | 模型唯一标识（自动填充） |
| `fake_prob` | float | 伪造概率 [0, 1] |
| `label` | str | "fake" / "real" / "uncertain" |
| `confidence` | float | 模型置信度 [0, 1] |
| `latency_ms` | float | 推理耗时（自动填充） |
| `details` | dict | 模型特有附加信息 |
| `error` | str/None | 错误信息（失败时填充） |

#### BaseDetector 工具方法

- `self._softmax(logits, axis=-1)`：数值稳定的 softmax
- `self._sigmoid(x)`：sigmoid
- `self.predict(img)`：不要重写，基类已处理计时和异常捕获
- `self._predict(img)`：必须实现，实际推理逻辑

## 3. 权重准备

### 3.1 ONNX 导出要求

- 输入维度：`(batch, 3, H, W)`，NCHW 格式，float32
- 输出维度：分类模型 `(batch, num_classes)` 或 `(batch, 1)`
- 动态 batch 轴：导出时设置 `dynamic_axes`

### 3.2 使用转换脚本

如果模型是 PyTorch 且结构类似已有模型，可扩展 `scripts/convert_to_onnx.py`：

```python
def convert_my_model(checkpoint: str, output: str, opset: int = 17) -> None:
    import torch
    # 1. 构建模型结构
    model = MyModel()
    # 2. 加载权重
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    # 3. 导出
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, output,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

# 注册到 CONVERTERS
CONVERTERS["my_model"] = convert_my_model
```

### 3.3 验证 ONNX 模型

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("my_detector.onnx")
print("Input:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
print("Output:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

# 测试推理
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = sess.run(None, {"input": dummy})
print("Result shape:", result[0].shape)
```

## 4. 配置启用

### 4.1 修改 .env

```env
ENABLED_MODELS=my_detector,cnn_spotter,freq_detector
```

### 4.2 设置融合权重（可选）

在 `FusionEngine` 初始化时传入自定义权重：

```python
fusion = FusionEngine(
    strategy="weighted_vote",
    model_weights={
        "my_detector": 1.5,      # 提高新模型权重
        "cnn_spotter": 1.0,
        "freq_detector": 0.8,
    }
)
```

## 5. 测试验证

### 5.1 单元测试

```python
# tests/test_my_detector.py
from PIL import Image
from app.models.my_detector import MyDetector

def test_predict():
    model = MyDetector(weights_dir="./models/weights")
    model.load()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = model.predict(img)
    assert 0 <= result.fake_prob <= 1
    assert result.label in ("fake", "real", "uncertain")
    assert result.latency_ms > 0
```

### 5.2 集成测试

```bash
# 启动服务后测试
curl -X POST http://localhost/api/v1/detect \
  -F "file=@test_image.jpg" | jq .
```

检查响应中 `per_model` 是否包含新模型的结果。

### 5.3 性能测试

```bash
# 使用 ab 或 wrk 压测
ab -n 100 -c 10 -p image.jpg -T "multipart/form-data" \
  http://localhost/api/v1/detect
```

关注指标：
- P50 / P95 / P99 延迟
- 吞吐量（QPS）
- 模型推理耗时占比

## 6. 无权重模型集成

如果模型不需要深度学习权重（如纯统计方法），可以简化 `_load()`：

```python
def _load(self) -> None:
    # 无需加载权重
    self._loaded = True
```

参考 `app/models/freq_detector.py` 的实现。

## 7. 常见问题

### Q: 模型输出不是概率而是 logits？
A: 使用 `self._softmax(logits)` 转换，或 `self._sigmoid(logit)` 处理单输出。

### Q: 模型需要特殊预处理？
A: 重写预处理逻辑，不要直接使用 `resize_to_tensor`。参考 `compute_frequency_spectrum` 的做法。

### Q: 模型推理很慢？
A: 1) 确认使用了 ONNX 而非 PyTorch；2) 启用量化；3) 重写 `batch_predict` 利用批量推理；4) 检查是否使用了 GPU。

### Q: 权重加载失败但不想影响其他模型？
A: 框架已处理：`_load()` 抛出异常时，该模型被跳过，其他模型正常工作。日志中会记录 `model_load_failed`。

### Q: 如何处理不同尺寸的输入？
A: 在 `_predict` 中统一 resize 到模型要求的尺寸。`resize_to_tensor` 已封装此逻辑。
