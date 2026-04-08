import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.api import app


def _encode_test_image() -> bytes:
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def test_health_endpoint_returns_metadata():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "app_name" in payload
    assert "app_version" in payload


def test_predict_returns_compressed_mask(monkeypatch):
    client = TestClient(app)
    expected_mask = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)

    def fake_predict(_image):
        return expected_mask

    monkeypatch.setattr("backend.api.predict", fake_predict)

    response = client.post(
        "/predict",
        files={"file": ("test.png", _encode_test_image(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mask_encoding"] == "png_base64"
    mask_bytes = base64.b64decode(payload["mask_png_base64"])
    decoded_mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded_mask.shape == expected_mask.shape
    assert payload["image_width"] == 8
    assert payload["image_height"] == 8


def test_predict_rejects_unsupported_content_type():
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("test.gif", b"fake", "image/gif")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
