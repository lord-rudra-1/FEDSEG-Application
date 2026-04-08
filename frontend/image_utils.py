import cv2
import numpy as np


def decode_uploaded_image(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def apply_cmap_gradient(m, r0, g0, b0, r1, g1, b1):
    m = np.clip(m, 0, 1)
    r = (r0 + (r1 - r0) * m).astype(np.uint8)
    g = (g0 + (g1 - g0) * m).astype(np.uint8)
    b = (b0 + (b1 - b0) * m).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def mask_to_rgb(mask, cmap_name):
    m = np.clip(mask, 0, 1)
    if cmap_name == "Grayscale":
        u = (m * 255).astype(np.uint8)
        return cv2.cvtColor(u, cv2.COLOR_GRAY2RGB)
    if cmap_name == "Purple (medical)":
        rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (m * 165).astype(np.uint8)
        rgb[..., 1] = (m * 98).astype(np.uint8)
        rgb[..., 2] = (m * 237).astype(np.uint8)
        return rgb
    if cmap_name == "Green (medical)":
        rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
        rgb[..., 1] = (m * 255).astype(np.uint8)
        rgb[..., 0] = (m * 136).astype(np.uint8)
        rgb[..., 2] = (m * 140).astype(np.uint8)
        return rgb
    if cmap_name == "Viridis":
        return apply_cmap_gradient(
            m,
            np.full_like(m, 68),
            np.full_like(m, 1),
            np.full_like(m, 84),
            np.full_like(m, 253),
            np.full_like(m, 231),
            np.full_like(m, 36),
        )
    if cmap_name == "Plasma":
        return apply_cmap_gradient(
            m,
            np.full_like(m, 13),
            np.full_like(m, 8),
            np.full_like(m, 135),
            np.full_like(m, 240),
            np.full_like(m, 249),
            np.full_like(m, 33),
        )
    u = (m * 255).astype(np.uint8)
    return cv2.cvtColor(u, cv2.COLOR_GRAY2RGB)


def make_overlay(original_rgb, mask_resized, threshold, alpha=0.45, tint=None):
    overlay = original_rgb.astype(np.float32).copy()
    tint = np.array([14, 116, 216], dtype=np.float32) if tint is None else np.array(tint, dtype=np.float32)
    foreground = mask_resized > threshold
    for channel in range(3):
        overlay[:, :, channel] = np.where(
            foreground,
            (1 - alpha) * overlay[:, :, channel] + alpha * tint[channel],
            overlay[:, :, channel],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_comparison_split(original_rgb, overlay_rgb, split_pct):
    split = int(original_rgb.shape[1] * split_pct)
    compare = original_rgb.copy()
    compare[:, split:, :] = overlay_rgb[:, split:, :]
    return compare
