import base64

import cv2
import numpy as np

from frontend.api_client import decode_mask_png_base64
from frontend.image_utils import make_comparison_split, make_overlay, mask_to_rgb


def test_decode_mask_png_base64_round_trip():
    original = np.array([[0, 127], [255, 64]], dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", original)
    assert ok

    decoded = decode_mask_png_base64(base64.b64encode(buffer.tobytes()).decode("ascii"))

    assert decoded.shape == (2, 2)
    assert np.isclose(decoded[0, 0], 0.0)
    assert np.isclose(decoded[1, 0], 1.0)


def test_mask_to_rgb_returns_three_channels():
    mask = np.array([[0.1, 0.9], [0.5, 0.7]], dtype=np.float32)

    rgb = mask_to_rgb(mask, "Viridis")

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8


def test_overlay_and_split_preserve_shape():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:, 2:] = 1.0

    overlay = make_overlay(original, mask, threshold=0.5, alpha=0.5, tint=[255, 0, 0])
    compare = make_comparison_split(original, overlay, 0.5)

    assert overlay.shape == original.shape
    assert compare.shape == original.shape
