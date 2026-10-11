"""
API 集成测试（使用 FastAPI TestClient，无需真实 Redis/模型权重）。
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def client():
    """创建测试用 FastAPI 客户端（mock 模型 + 内存缓存）。"""
    from fastapi.testclient import TestClient
    from app.main import create_app

    # 强制使用 mock 模型
    import os
    os.environ["ENABLED_MODELS"] = "mock"
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    os.environ["CACHE_TTL"] = "60"

    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_image_bytes(color=(128, 128, 128), size=(100, 100)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert data["models_loaded"] >= 1


class TestDetectEndpoint:
    def test_detect_png(self, client):
        img_bytes = _make_image_bytes()
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "fake_prob" in data
        assert 0 <= data["fake_prob"] <= 1
        assert data["label"] in ("fake", "real", "uncertain")
        assert data["total_latency_ms"] > 0
        assert len(data["per_model"]) >= 1

    def test_detect_jpeg(self, client):
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["postprocessing_type"] == "jpeg"

    def test_invalid_file(self, client):
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_empty_file(self, client):
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code in (400, 422)


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled_count"] >= 1
        assert len(data["models"]) >= 1
        assert data["models"][0]["model_id"] == "mock"


class TestCacheEndpoint:
    def test_clear_cache(self, client):
        resp = client.post("/api/v1/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
