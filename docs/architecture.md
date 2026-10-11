# 架构设计文档

## 1. 整体架构

采用经典的**三层架构**，层间通过明确接口解耦：

```
                    ┌──────────────────┐
                    │   客户端 / 业务方  │
                    └────────┬─────────┘
                             │ HTTPS
                    ┌────────▼─────────┐
                    │   Nginx 反向代理  │  静态资源 + 负载均衡
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │      接入网关层 (Gateway)     │
              │  FastAPI · 鉴权 · 限流 · 校验 │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      推理引擎层 (Engine)      │
              │  缓存查询 → 并行推理 → 融合判决 │
              └──────┬───────────┬───────────┘
                     │           │
              ┌──────▼───┐  ┌────▼──────┐
              │ 模型层    │  │  Redis    │
              │ 多模型    │  │ 缓存+限流  │
              └──────────┘  └───────────┘
```

## 2. 接入网关层

### 职责
- 统一 REST API 入口
- API Key 鉴权（可选）
- 请求参数校验（图片格式、大小）
- 分布式限流（令牌桶）
- 请求上下文（request_id）
- 全局异常处理

### 关键设计

**API Key 鉴权**：作为 FastAPI 依赖注入，未配置 `API_KEY` 时自动跳过（内网部署场景）。

**令牌桶限流**：
- 基于 Redis Lua 脚本实现，多实例共享状态
- Redis 不可用时降级为进程内令牌桶
- 按客户端 IP 维度限流，返回 429 + Retry-After

**请求生命周期**：
```
请求到达 → 生成 request_id → 鉴权 → 限流 → 图片校验
→ 缓存查询 → 推理 → 缓存写入 → 响应（含 X-Request-ID）
```

## 3. 推理引擎层

### 3.1 推理流水线 (InferencePipeline)

```
┌──────────────────────────────────────────────────┐
│              detect(image, cache_key)             │
├──────────────────────────────────────────────────┤
│  1. 缓存查询 ──命中──→ 直接返回（<1ms）            │
│       │未命中                                      │
│  2. 后处理类型检测（JPEG / 缩放）                   │
│       │                                           │
│  3. 并行推理（asyncio.gather）                     │
│       ├── Model A ──→ result_A                    │
│       ├── Model B ──→ result_B                    │
│       └── Model C ──→ result_C (timeout→降级)     │
│       │                                           │
│  4. 融合判决（FusionEngine）                       │
│       │                                           │
│  5. 结果写入缓存                                   │
│       │                                           │
│  6. 指标上报 + 返回                                │
└──────────────────────────────────────────────────┘
```

### 3.2 多模型融合判决 (FusionEngine)

支持四种策略：

| 策略 | 算法 | 适用场景 |
|------|------|----------|
| `weighted_vote` | Σ(fake_prob × weight) / Σ(weight) | 通用场景，平衡各模型（默认） |
| `max_prob` | max(fake_prob) | 严格审核，降低漏报 |
| `confidence_weighted` | Σ(fake_prob × weight × confidence) | 模型置信度差异大时 |
| `adaptive` | 检测模型组分歧，分歧大时降级 uncertain | 模型一致性差时 |

**三态判决区间**：

| fake_prob | 标签 | 说明 |
|-----------|------|------|
| ≤ 0.35 | `real` | 判定真实 |
| 0.35 ~ 0.7 | `uncertain` | 模型分歧大，建议人工复核 |
| ≥ 0.7 | `fake` | 判定 AI 生成 |

通过 `UNCERTAIN_LOW` / `UNCERTAIN_HIGH` 环境变量可调。

**动态权重调整**：检测到图片经过 JPEG 压缩时，对 `strong_postprocessing` 包含 "jpeg" 的模型权重 ×1.2。

**融合置信度**：综合模型间一致性（1 - 标准差）和最终概率偏离度。

### 3.3 异步批量推理 (BatchProcessor)

```
请求1 ──┐
请求2 ──┼──→ Queue ──→ 攒批(max_size=8, timeout=50ms) ──→ batch_predict() ──→ 分发结果
请求3 ──┘
```

- 每个模型独立一个 BatchProcessor
- 达到 batch_max_size 或等待 batch_timeout_ms 后 flush
- 批量推理失败时自动降级为逐张单图推理
- 通过 asyncio.Future 实现请求-结果配对

### 3.4 容错设计

- **单模型超时**：`asyncio.wait_for` 控制，超时返回 uncertain，不阻断融合
- **单模型异常**：try/except 捕获，记录 error 字段，融合时跳过
- **全部失败**：返回 label=uncertain, confidence=0，由业务方决策
- **Redis 故障**：缓存降级为内存 LRU，限流器降级为进程内

## 4. 模型层

### 4.1 统一抽象 (BaseDetector)

```python
class BaseDetector(ABC):
    model_id: str           # 唯一标识
    model_name: str         # 显示名称
    version: str            # 版本
    default_weight: float   # 融合默认权重
    strong_postprocessing: list  # 擅长的后处理类型

    def load(self) -> None          # 加载权重（幂等）
    def predict(self, img) -> DetectionResult  # 单图推理（含计时+异常捕获）
    def batch_predict(self, imgs) -> list      # 批量推理（可重写）
```

所有模型适配器继承此类，上层引擎只依赖接口，不关心具体实现。

### 4.2 模型注册中心 (ModelRegistry)

- 装饰器模式 `@register_model("model_id")`
- 自动发现：`auto_discover()` 导入所有模型模块
- 支持运行时动态注册新模型

### 4.3 已集成模型（8款 + Mock）

**无权重模型（5款，开箱即用，多原理互补）：**

| 模型 | 原理 | 默认权重 | 延迟 | 特点 |
|------|------|---------|------|------|
| FreqDetector | 频域对数幅度谱统计 | 0.3 | ~30ms | 零依赖、极快、JPEG 鲁棒 |
| ELADetector | JPEG 重压缩误差分析 | 0.7 | ~35ms | 检测压缩均匀性异常 |
| NoiseResidualDetector | 中值滤波噪声残差+自相关 | 0.6 | ~35ms | cv2.medianBlur 优化，识别上采样伪影 |
| CFADetector | Bayer CFA 插值伪影频谱 | 0.2 | ~60ms | 真实照片特有周期性，AI 生成缺失 |
| ColorDistributionDetector | HSV 色彩分布统计 | 0.8 | ~70ms | 饱和度/色相熵/色彩丰富度异常 |

**深度学习模型（3款，ONNX 格式，高权重主导判决）：**

| 模型 | 原理 | 默认权重 | 延迟 | 权重大小 |
|------|------|---------|------|---------|
| CNNSpotter | ResNet50 微调（CNNDetection） | 3.0 | ~100ms | 90MB |
| UniversalFake | CLIP ViT-L/14 + 线性头 | 3.0 | ~1000ms | 1.16GB |
| AeroBlade | SD VAE 重建误差（CVPR 2024） | 1.0 | ~1100ms | 320MB |

另有 Mock 模型用于开发测试。深度学习模型权重（3.0）远高于无权重模型（0.2~0.8），主导融合判决。无权重模型作为辅助信号，对压缩、裁剪等后处理提供鲁棒性补充。

### 4.4 ONNX 推理优化

- `GraphOptimizationLevel.ORT_ENABLE_ALL`：启用全部图优化
- 传统 TorchScript 导出器（dynamo=False）：ONNX Runtime 1.19 兼容性更好
- 固定输入尺寸导出：比动态尺寸推理更快（AeroBlade 128×128 固定）
- 自动检测 CUDA ExecutionProvider，有 GPU 时自动使用
- 动态 batch 轴：支持批量推理

## 5. 缓存设计

### 缓存键
- 主键：`img:{md5(raw_bytes)}` —— 精确匹配完全相同的图片
- 预留：pHash 感知哈希用于近似命中（后续可扩展）

### 缓存内容
```json
{
  "fusion": { ...融合结果... },
  "postprocessing_type": "jpeg"
}
```

### 降级策略
Redis 不可用时自动使用进程内 LRU 缓存（默认 10000 条），保证服务可用性。

## 6. 可观测性

### 指标 (Prometheus)
| 指标名 | 类型 | 标签 | 说明 |
|--------|------|------|------|
| `auth_gateway_requests_total` | Counter | endpoint, status | 请求总数 |
| `auth_gateway_request_duration_seconds` | Histogram | endpoint | 请求耗时 |
| `auth_gateway_inference_duration_seconds` | Histogram | model | 单模型推理耗时 |
| `auth_gateway_detections_total` | Counter | label | 检测结果分布 |
| `auth_gateway_cache_hits_total` | Counter | - | 缓存命中 |
| `auth_gateway_cache_misses_total` | Counter | - | 缓存未命中 |
| `auth_gateway_rate_limited_total` | Counter | - | 限流拒绝 |
| `auth_gateway_model_errors_total` | Counter | model | 模型错误 |

### 日志 (structlog)
- JSON 格式，包含 request_id、method、path、status、duration_ms
- 每个请求绑定 request_id，便于全链路追踪
- 模型加载失败、推理超时等异常均有结构化日志

### 健康检查
`GET /health` 返回：
- 服务状态（ok / degraded）
- 已加载模型数 / 总模型数
- Redis 可用性
- 运行时长

## 7. 扩展点

### 新增检测模型
1. 继承 `BaseDetector`，实现 `_load()` 和 `_predict()`
2. 添加 `@register_model("new_model")` 装饰器
3. 在 `registry.auto_discover()` 中注册模块
4. `.env` 中 `ENABLED_MODELS` 添加新模型 ID

### 新增融合策略
1. 在 `FusionEngine.VALID_STRATEGIES` 中添加
2. 实现对应的 `_xxx()` 方法
3. `.env` 中 `FUSION_STRATEGY` 切换

### 文本/视频检测
- 文本：新增 `TextDetector` 基类，复用融合引擎
- 视频：抽帧 → 逐帧图像检测 → 时序聚合（投票/平滑）
