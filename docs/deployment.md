# 部署指南

## 1. Docker Compose 部署（推荐）

### 前置要求
- Docker 20.10+
- Docker Compose v2
- 至少 2GB 可用内存（4GB 推荐）

### 部署步骤

```bash
# 1. 进入项目目录
cd ai-content-authenticity-gateway

# 2. 复制并编辑环境变量
cp .env.example .env
vim .env

# 3. 构建并启动
docker compose up -d --build

# 4. 查看状态
docker compose ps
docker compose logs -f gateway

# 5. 健康检查
curl http://localhost/health
```

### 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| Nginx | 80 | 前端 + API 反向代理 |
| Gateway | 8000 | FastAPI（容器内部，不直接暴露） |
| Redis | 6379 | 缓存（容器内部，不直接暴露） |

### 常用运维命令

```bash
# 查看日志
docker compose logs -f gateway
docker compose logs -f redis

# 重启服务
docker compose restart gateway

# 停止并移除容器
docker compose down

# 停止并移除容器 + 数据卷（清空 Redis 数据）
docker compose down -v

# 重新构建（代码变更后）
docker compose up -d --build gateway
```

## 2. 裸机部署（无 Docker）

### 2.1 安装依赖

```bash
# Python 3.11+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Redis（Ubuntu/Debian）
sudo apt-get install redis-server
sudo systemctl start redis-server

# Redis（macOS）
brew install redis
brew services start redis

# Redis（Windows）
# 使用 WSL 或 Memurai，或设置 CACHE_TTL 仅用内存缓存
```

### 2.2 配置环境变量

```bash
cp .env.example .env
# 修改 Redis 地址为 localhost
# .env: REDIS_HOST=localhost
```

### 2.3 启动服务

```bash
# 开发模式（热重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式（多 worker）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2.4 配置 Nginx（可选）

```bash
sudo cp nginx/nginx.conf /etc/nginx/conf.d/auth-gateway.conf
# 修改 upstream 中的 server 地址为 127.0.0.1:8000
sudo nginx -t
sudo systemctl reload nginx
```

### 2.5 配置 Systemd 服务（生产推荐）

```ini
# /etc/systemd/system/auth-gateway.service
[Unit]
Description=AI Content Authenticity Gateway
After=network.target redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/auth-gateway
Environment="PATH=/opt/auth-gateway/venv/bin"
ExecStart=/opt/auth-gateway/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now auth-gateway
```

## 3. Podman 部署

```bash
# Podman 兼容 Docker Compose（需安装 podman-compose）
pip install podman-compose
podman-compose up -d

# 或使用 podman 直接运行
podman network create auth-net
podman run -d --name auth-redis --network auth-net redis:7-alpine
podman build -t auth-gateway .
podman run -d --name auth-gateway --network auth-net \
  -e REDIS_HOST=auth-redis \
  -p 8000:8000 auth-gateway
```

## 4. 模型权重部署

项目包含两类模型：
- **无权重模型**（5款）：freq_detector, ela_detector, noise_detector, cfa_detector, color_detector —— 开箱即用，无需下载
- **深度学习模型**（3款）：cnn_spotter, universal_fake, aeroblade —— 需下载权重并转换为 ONNX

### 4.1 无权重模型（默认启用）

无需任何操作，`.env` 默认已配置：
```env
ENABLED_MODELS=freq_detector,ela_detector,noise_detector,cfa_detector,color_detector
```

### 4.2 深度学习模型权重获取

项目提供 3 款深度学习模型的转换脚本，需在有外网（或镜像站）的环境运行：

#### CNNDetection + UniversalFakeDetect

```bash
python scripts/convert_deep_models.py
```
脚本自动：安装依赖 → 从 Dropbox 下载 CNNDetection 权重 → 从 GitHub 下载 UniversalFake 权重 → 转换为 ONNX → 验证推理。

> 国内用户：Dropbox 可能需要代理。可使用 Colab 笔记本（`scripts/colab_convert_weights.ipynb`）在线转换后下载。

#### AeroBlade (SD VAE)

```bash
# 国内用户先设置镜像站
$env:HF_ENDPOINT="https://hf-mirror.com"   # Windows (PowerShell)
# export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS

python scripts/convert_aeroblade.py
```
脚本自动：从 HuggingFace 下载 Stable Diffusion 1.5 VAE → 导出编码器/解码器 ONNX（固定 128×128 输入）→ 验证端到端推理。

输出文件：
- `aeroblade_encoder.onnx` + `.data`（130MB）
- `aeroblade_decoder.onnx` + `.data`（189MB）

### 4.3 量化压缩（可选但推荐）

```bash
# 动态量化（无需校准数据，体积减少 ~75%）
python scripts/quantize.py \
  --input models/weights/cnn_spotter.onnx \
  --output models/weights/cnn_spotter.onnx \
  --method dynamic

# 静态量化（需要校准图片，精度更高）
python scripts/quantize.py \
  --input models/weights/cnn_spotter.onnx \
  --output models/weights/cnn_spotter_int8.onnx \
  --method static --calib-dir ./calib_images
```

### 4.4 启用模型

```env
# .env - 启用全部 8 款模型
ENABLED_MODELS=freq_detector,ela_detector,noise_detector,cfa_detector,color_detector,cnn_spotter,universal_fake,aeroblade
FUSION_STRATEGY=weighted_vote
```

Docker 部署时权重目录通过 volume 挂载，无需重新构建镜像：
```yaml
volumes:
  - ./models/weights:/app/models/weights:ro
```

> **注意**：权重加载失败的模型会被自动跳过，不影响其他模型正常工作。可通过 `GET /api/v1/models` 查看已加载模型。

## 5. 性能调优

### 5.1 Worker 数量

```env
# CPU 密集型推理，worker 数 = CPU 核心数
APP_WORKERS=4
```

### 5.2 批量推理参数

```env
# 高并发场景：增大 batch，减少 flush 频率
BATCH_MAX_SIZE=16
BATCH_TIMEOUT_MS=30

# 低延迟场景：减小 batch，快速 flush
BATCH_MAX_SIZE=4
BATCH_TIMEOUT_MS=10
```

### 5.3 限流参数

```env
# 根据后端处理能力调整
RATE_LIMIT_RATE=20      # 每秒 20 个请求
RATE_LIMIT_CAPACITY=40  # 突发 40 个
```

### 5.4 ONNX 线程

在模型适配器中调整 `intra_op_num_threads`：
```python
sess_options.intra_op_num_threads = 4  # 根据 CPU 核心数
```

### 5.5 GPU 加速

如果部署环境有 NVIDIA GPU：
1. 使用 `nvidia/cuda` 基础镜像修改 Dockerfile
2. 安装 `onnxruntime-gpu` 替代 `onnxruntime`
3. 模型适配器会自动检测并使用 CUDA ExecutionProvider

### 5.6 模型推理优化实践

| 优化项 | 方法 | 效果 |
|--------|------|------|
| AeroBlade 分辨率 | 固定 128×128 输入（修改 `convert_aeroblade.py` 中 dummy 尺寸） | 7.6s → 1.1s |
| noise_detector | `cv2.medianBlur` 替代手动双重循环 | 823ms → 36ms |
| ONNX 导出器 | 使用 `dynamo=False` 传统导出器 | ONNX Runtime 1.19 兼容性更好 |
| 固定输入尺寸 | 导出时不设置动态空间维度 | 比动态尺寸推理快 2-3x |
| 模型并行 | `asyncio.gather` 并发推理 | 总延迟 = 最慢模型，非累加 |

> AeroBlade 分辨率修改后需重新运行 `convert_aeroblade.py` 导出新 ONNX，并同步修改 `app/models/aeroblade.py` 中的 `INPUT_SIZE`。

## 6. 监控与告警

### 6.1 Prometheus 抓取

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'auth-gateway'
    static_configs:
      - targets: ['localhost:80']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### 6.2 关键告警规则

| 告警 | 条件 | 级别 |
|------|------|------|
| 服务不可用 | `up == 0` 持续 1m | P0 |
| 高错误率 | `rate(requests_total{status=~"5xx"}[5m]) > 0.05` | P1 |
| 高延迟 | `histogram_quantile(0.95, request_duration_seconds) > 5s` | P1 |
| 模型错误激增 | `rate(model_errors_total[5m]) > 10` | P2 |
| 限流严重 | `rate(rate_limited_total[5m]) > 5` | P2 |

### 6.3 日志采集

日志输出为 JSON 到 stdout，可通过 Docker logging driver 采集：
```yaml
# docker-compose.yml 中添加
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "3"
```

配合 Filebeat / Fluentd 采集到 ELK / Loki。

## 7. 安全加固

### 7.1 API Key

```env
# .env 中设置强密钥
API_KEY=your-strong-api-key-here
```

调用时在 Header 中携带：
```
X-API-Key: your-strong-api-key-here
```

### 7.2 Nginx 安全

- 已配置 `X-Content-Type-Options`、`X-Frame-Options`、`X-XSS-Protection`
- `/metrics` 端点建议加 IP 白名单（nginx.conf 中有注释示例）
- 生产环境建议配置 HTTPS（Let's Encrypt / 内部 CA）

### 7.3 网络隔离

- Redis 不暴露到宿主机端口（仅容器内部网络）
- Gateway 不直接暴露，通过 Nginx 反向代理
- 建议部署在内网，通过 VPN / 堡垒机访问

## 8. 故障排查

### 服务启动失败

```bash
# 查看详细日志
docker compose logs gateway --tail 100

# 常见原因：
# 1. 端口被占用 → 修改 docker-compose.yml 的端口映射
# 2. 内存不足 → 增加 deploy.resources.limits.memory
# 3. 模型权重缺失 → 检查 models/weights/ 目录
```

### 检测结果全部为 uncertain

- 检查模型是否加载成功：`curl http://localhost/api/v1/models`
- 查看模型错误日志
- 确认权重文件格式正确（ONNX）

### Redis 连接失败

- 确认 Redis 容器运行：`docker compose ps redis`
- 检查 `.env` 中 `REDIS_HOST`（Docker 环境应为 `redis`）
- 服务会自动降级为内存缓存，不影响推理功能

### 推理超时

- 增大 `INFERENCE_TIMEOUT`（默认 30s）
- 检查图片尺寸是否过大
- 考虑启用量化模型减小推理耗时
- GPU 环境确认 CUDA 可用
