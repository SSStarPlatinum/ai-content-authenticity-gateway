# AI 生成内容鉴真私有化推理网关

> 企业级 AI 生成图像伪造检测推理服务 —— 私有化部署、多模型融合、高性能推理、全链路可观测。

## 项目定位

本项目将零散的 AI 伪造检测开源模型，封装为企业级可落地的私有化推理服务。核心不是发明新算法，而是**工程化落地**：统一接口、性能优化、多模型融合、可观测运维。

### 解决的核心痛点

| 痛点 | 解决方案 |
|------|----------|
| 公有云检测 API 存在数据隐私风险 | 全私有化部署，数据不出内网 |
| 开源模型接口不统一、部署门槛高 | 统一 BaseDetector 抽象 + 标准化 REST API |
| 单一模型泛化性弱、后处理鲁棒性差 | 8 款不同原理模型加权融合 + 三态判决区间 |
| 缺乏推理链路可观测性 | Prometheus 指标 + 结构化日志 + 健康检查 |
| 高并发下吞吐不足 | ONNX 加速 + 图片哈希缓存 + 异步批量推理 |

## 三层架构

```
┌─────────────────────────────────────────────────┐
│              接入网关层 (Gateway)                 │
│  FastAPI · Nginx · API Key · 限流 · CORS         │
├─────────────────────────────────────────────────┤
│              推理引擎层 (Engine)                  │
│  融合判决 · 批量推理 · 流水线编排 · 超时控制       │
├─────────────────────────────────────────────────┤
│              模型层 (Models)                      │
│  无权重: Freq·ELA·Noise·CFA·Color                 │
│  深度学习: CNNSpotter·UniversalFake·AeroBlade     │
│  ONNX Runtime · 统一预处理 · 模型注册中心         │
└─────────────────────────────────────────────────┘
         ↕ Redis (缓存 + 分布式限流)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端网关 | Python 3.11 · FastAPI · Uvicorn |
| 推理加速 | ONNX Runtime（可扩展 TensorRT） |
| 缓存与限流 | Redis（分布式令牌桶） |
| 部署运维 | Docker / Podman · Docker Compose · Nginx |
| 可观测性 | Prometheus · structlog（JSON 日志） |
| 前端 | 纯 HTML + JS（无构建依赖） |
| 测试 | pytest · FastAPI TestClient |

## 已集成模型（8 款）

### 无权重模型（5 款，开箱即用）

| 模型 ID | 原理 | 默认权重 | 推理延迟 |
|---------|------|---------|---------|
| `freq_detector` | 频域对数幅度谱统计 | 0.3 | ~30ms |
| `ela_detector` | JPEG 重压缩误差分析 | 0.7 | ~35ms |
| `noise_detector` | 中值滤波噪声残差+自相关 | 0.6 | ~35ms |
| `cfa_detector` | Bayer CFA 插值伪影频谱 | 0.2 | ~60ms |
| `color_detector` | HSV 色彩分布统计 | 0.8 | ~70ms |

### 深度学习模型（3 款，需 ONNX 权重）

| 模型 ID | 原理 | 默认权重 | 推理延迟 | 权重大小 |
|---------|------|---------|---------|---------|
| `cnn_spotter` | ResNet50 微调（CNNDetection） | 3.0 | ~100ms | 90MB |
| `universal_fake` | CLIP ViT-L/14 + 线性头 | 3.0 | ~1000ms | 1.16GB |
| `aeroblade` | SD VAE 重建误差（CVPR 2024） | 1.0 | ~1100ms | 320MB |

> 深度学习模型权重高（3.0），主导融合判决；无权重模型作为辅助信号提升鲁棒性。

## 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
cd ai-content-authenticity-gateway

# 2. 配置环境变量
cp .env.example .env
# 默认启用 5 款无权重模型，可直接启动

# 3. 一键启动
docker compose up -d

# 4. 访问
# 前端演示页: http://localhost
# API 文档:   http://localhost:8000/docs
# 健康检查:   http://localhost/health
```

### 方式二：启用全部 8 款模型

```bash
# 1. 下载并转换深度学习模型权重（需外网或镜像站）
# CNNDetection + UniversalFakeDetect:
python scripts/convert_deep_models.py
# AeroBlade (SD VAE):
$env:HF_ENDPOINT="https://hf-mirror.com"  # Windows，国内用户
python scripts/convert_aeroblade.py

# 2. 修改 .env，启用全部模型
ENABLED_MODELS=freq_detector,ela_detector,noise_detector,cfa_detector,color_detector,cnn_spotter,universal_fake,aeroblade

# 3. 重建并启动
docker compose up -d --build
```

### 方式三：本地开发

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 接口

### 图片鉴真检测

```
POST /api/v1/detect
Content-Type: multipart/form-data
X-API-Key: <your-key>    # 可选，配置 API_KEY 后必填

file: <image binary>
```

**响应示例：**

```json
{
  "success": true,
  "fake_prob": 0.873,
  "label": "fake",
  "confidence": 0.746,
  "strategy": "weighted_vote",
  "total_latency_ms": 1150.5,
  "cache_hit": false,
  "model_count": 8,
  "failed_models": [],
  "per_model": [
    {
      "model_id": "cnn_spotter",
      "fake_prob": 0.91,
      "label": "fake",
      "confidence": 0.82,
      "latency_ms": 98.2,
      "details": {"input_size": "224x224"}
    }
  ],
  "image_meta": {"width": 1024, "height": 1024, "format": "JPEG"},
  "request_id": "a1b2c3d4-..."
}
```

### 三态判决

| fake_prob 区间 | 标签 | 含义 |
|---------------|------|------|
| ≤ 0.35 | `real` | 判定为真实图像 |
| 0.35 ~ 0.7 | `uncertain` | 模型分歧大，建议人工复核 |
| ≥ 0.7 | `fake` | 判定为 AI 生成 |

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/models` | 列出已加载模型 |
| GET | `/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |
| POST | `/api/v1/cache/clear` | 清空缓存 |

## 准确率基准

在 AIDE 数据集（Chameleon / face / progan，共 412 张）上的评估结果：

| 数据集 | 样本数 | 准确率 | 说明 |
|--------|--------|--------|------|
| progan | 292 | **100%** | GAN 生成图，深度学习模型完全覆盖 |
| face | 60 | 48.3% | 人脸生成图，22 张落入 uncertain |
| Chameleon | 60 | 48.3% | 文本生成图，深度学习模型泛化不足 |
| **总计** | **412** | **84.95%** | - |

> 各模型单独表现及优化细节见 `docs/architecture.md`。

## 性能优化

| 优化手段 | 效果 |
|---------|------|
| PyTorch → ONNX + 图优化 | 推理提速 2-3x |
| AeroBlade 输入 512→128 | 7.6s → 1.1s（提速 85%） |
| noise_detector 手动循环 → cv2.medianBlur | 823ms → 36ms（提速 96%） |
| 图片 MD5 哈希缓存 | 重复请求 <10ms |
| 多模型 asyncio.gather 并行 | 总延迟 = 最慢模型，而非累加 |
| 异步批量推理（攒批 8 张/50ms） | 高并发吞吐提升 |

**端到端延迟**：首次推理 ~1.1s（8 模型并行），缓存命中 <10ms。

## 项目结构

```
ai-content-authenticity-gateway/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── api/                 # 接入网关层
│   │   ├── routes.py        # API 路由
│   │   └── schemas.py       # 请求/响应 Schema
│   ├── gateway/             # 鉴权 + 限流
│   ├── engine/              # 推理引擎层
│   │   ├── fusion.py        # 多模型融合判决（含三态区间）
│   │   ├── batch.py         # 异步批量推理
│   │   └── pipeline.py      # 端到端推理流水线
│   ├── models/              # 模型层（8 款检测器）
│   │   ├── base.py          # 检测模型基类
│   │   ├── registry.py      # 模型注册中心
│   │   ├── freq_detector.py # 频域统计（无权重）
│   │   ├── ela_detector.py  # ELA 压缩误差（无权重）
│   │   ├── noise_detector.py# 噪声残差（无权重，cv2优化）
│   │   ├── cfa_detector.py  # CFA 插值伪影（无权重）
│   │   ├── color_detector.py# 色彩分布（无权重）
│   │   ├── cnn_spotter.py   # CNNDetection (ONNX)
│   │   ├── universal_fake.py# UniversalFakeDetect (ONNX)
│   │   └── aeroblade.py     # AeroBlade VAE (ONNX, 128x128)
│   ├── cache/redis_cache.py
│   ├── observability/       # 日志 + 指标 + 中间件
│   └── utils/               # 图像处理 + 哈希
├── frontend/                # 演示页（纯 HTML+JS）
├── scripts/                 # 工具脚本
│   ├── convert_deep_models.py   # CNN+UniversalFake 下载转换
│   ├── convert_aeroblade.py     # AeroBlade VAE 下载转换
│   ├── convert_to_onnx.py       # 通用 ONNX 转换
│   ├── evaluate_accuracy.py     # 数据集准确率评估
│   ├── benchmark_latency.py     # 推理延迟基准测试
│   ├── quantize.py              # ONNX 量化
│   ├── colab_convert_weights.ipynb  # Colab 在线转换
│   └── download_weights.ps1/sh      # 权重下载辅助
├── tests/                   # 单元测试 + 集成测试
├── docs/                    # 详细文档
├── nginx/nginx.conf         # Nginx 反向代理
├── Dockerfile               # 多阶段构建
├── docker-compose.yml       # 一键部署
├── requirements.txt
└── .env.example
```

## 运行测试

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## 扩展能力

- **文本检测**：新增 `TextDetector` 继承 `BaseDetector`，注册到 registry 即可
- **视频帧检测**：复用图像检测流水线，增加抽帧节点
- **TensorRT 加速**：ONNX → TensorRT engine，替换 `providers` 配置
- **新增融合策略**：在 `FusionEngine` 中实现，`.env` 切换

## 文档索引

- [架构设计](docs/architecture.md) — 三层架构、融合策略、容错设计
- [部署指南](docs/deployment.md) — Docker/裸机/Podman 部署、权重获取、性能调优
- [模型集成指南](docs/model_integration.md) — 新增模型适配器、ONNX 导出、测试验证
- [模型权重说明](models/README.md) — 权重文件清单与获取方式

## License

MIT
