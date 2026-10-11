# 更新日志

## [Unreleased]

## [1.0.0] - 2026-10-11

### 架构
- 三层架构：接入网关层（FastAPI+Nginx）、推理引擎层（融合+批量+流水线）、模型层（统一抽象+注册中心）
- 统一 `BaseDetector` 抽象，`@register_model` 装饰器自动发现
- 多模型融合引擎，支持 `weighted_vote` / `max_prob` / `confidence_weighted` / `adaptive` 四种策略
- 三态判决区间：real(≤0.35) / uncertain(0.35~0.7) / fake(≥0.7)

### 模型（8 款）
- **无权重（5 款）**：freq_detector、ela_detector、noise_detector、cfa_detector、color_detector
- **深度学习（3 款）**：cnn_spotter（CNNDetection/ResNet50）、universal_fake（CLIP ViT-L/14）、aeroblade（SD VAE 重建误差，CVPR 2024）
- 深度学习模型权重 3.0，主导融合判决；无权重模型 0.2~0.8，作为辅助信号

### 性能优化
- AeroBlade 输入分辨率 512→128，推理 7.6s→1.1s
- noise_detector 中值滤波手动循环→cv2.medianBlur，823ms→36ms
- ONNX 传统导出器（dynamo=False）+ 固定输入尺寸，兼容 ONNX Runtime 1.19
- 图片 MD5 哈希缓存，重复请求 <10ms
- 多模型 asyncio.gather 并行推理
- 异步批量推理（攒批 8 张/50ms）

### 工程化
- Docker Compose 一键部署（gateway + redis + nginx）
- Redis 分布式令牌桶限流，故障降级为进程内
- Prometheus 指标 + structlog JSON 日志
- API Key 鉴权（可选）
- 前端纯 HTML+JS 演示页，实时显示各模型分数与延迟

### 准确率基准（AIDE 数据集，412 张）
- progan: 100%（292 张）
- face: 48.3%（60 张，22 张 uncertain）
- Chameleon: 48.3%（60 张）
- 总计: 84.95%

### 工具脚本
- `convert_deep_models.py`：CNNDetection + UniversalFake 下载转换
- `convert_aeroblade.py`：SD VAE 下载转换（支持 hf-mirror 镜像）
- `evaluate_accuracy.py`：数据集批量准确率评估
- `benchmark_latency.py`：各模型推理延迟基准
- `colab_convert_weights.ipynb`：Colab 在线转换笔记本
