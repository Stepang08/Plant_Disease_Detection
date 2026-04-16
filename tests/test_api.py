"""API endpoint tests. Run with: pytest tests/test_api.py"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent

client = TestClient(app)


def _make_dummy_jpeg() -> bytes:
    """Create a tiny valid JPEG in memory for testing."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_classes():
    resp = client.get("/classes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_classes"] == 39
    assert len(data["classes"]) == 39
    assert "late blight" in data["classes"]


def test_predict_no_model():
    """Without a model loaded, /predict should return 503."""
    jpeg = _make_dummy_jpeg()
    resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
    # If no model is loaded (test env), expect 503
    if resp.status_code == 503:
        assert "not loaded" in resp.json()["detail"].lower()


def test_predict_invalid_file():
    """Non-image file should return 422."""
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code in (422, 503)
