import os
import cv2
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API_URL = os.environ.get("FEDSEG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FEDSEG AI Dashboard",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-0: #f6f8fc;
        --bg-1: #eef2ff;
        --bg-2: #e9f0ff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --card-bg: rgba(255, 255, 255, 0.68);
        --card-border: rgba(148, 163, 184, 0.26);
        --primary: #4f46e5;
        --primary-soft: #6366f1;
        --accent: #06b6d4;
        --success: #10b981;
        --danger: #ef4444;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background:
            radial-gradient(800px 350px at 10% 0%, rgba(79, 70, 229, 0.14), transparent 60%),
            radial-gradient(700px 300px at 90% 0%, rgba(6, 182, 212, 0.12), transparent 60%),
            linear-gradient(165deg, var(--bg-0) 0%, var(--bg-1) 48%, var(--bg-2) 100%);
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
    }

    .glass {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
    }

    .topbar {
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .topbar-badges {
        margin-top: 0.7rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .pill {
        font-size: 0.74rem;
        font-weight: 600;
        color: #1e293b;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 999px;
        padding: 0.26rem 0.6rem;
    }

    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #1e1b4b;
        margin: 0;
    }

    .brand-sub {
        margin: 0.15rem 0 0 0;
        color: var(--text-secondary);
        font-size: 0.92rem;
    }

    .status-wrap {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 600;
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    .section-card {
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .mini-stat {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(255,255,255,0.72);
        padding: 0.65rem 0.75rem;
        min-height: 82px;
    }
    .mini-stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin: 0;
    }
    .mini-stat-value {
        margin: 0.25rem 0 0 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e1b4b;
    }

    .section-title {
        font-size: 1.04rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
        color: #1f2937;
    }

    .section-subtitle {
        font-size: 0.86rem;
        color: var(--text-secondary);
        margin-bottom: 0.9rem;
    }

    .metric-card {
        padding: 0.95rem 1rem;
        border-radius: 14px;
        background: linear-gradient(170deg, rgba(255,255,255,0.78) 0%, rgba(255,255,255,0.55) 100%);
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.10);
        transition: transform 150ms ease, box-shadow 150ms ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.18);
    }

    .metric-label {
        margin: 0;
        font-size: 0.73rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }

    .metric-value {
        margin: 0.35rem 0 0 0;
        font-size: 1.45rem;
        font-weight: 800;
        color: #1e1b4b;
        letter-spacing: -0.02em;
    }

    .upload-hint {
        border: 1px dashed rgba(79, 70, 229, 0.35);
        background: rgba(99, 102, 241, 0.06);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        color: #4338ca;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .empty-state {
        text-align: center;
        padding: 2.1rem 1rem;
        border-radius: 16px;
        border: 1px dashed rgba(100, 116, 139, 0.36);
        background: rgba(255, 255, 255, 0.62);
        color: #334155;
    }

    .empty-title {
        font-weight: 700;
        font-size: 1.06rem;
        margin-bottom: 0.3rem;
        color: #1f2937;
    }

    .empty-sub {
        margin: 0;
        font-size: 0.9rem;
        color: #64748b;
    }
    .step-row {
        display: flex;
        justify-content: center;
        gap: 0.55rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    .step-chip {
        font-size: 0.74rem;
        border-radius: 999px;
        padding: 0.2rem 0.62rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        background: rgba(255,255,255,0.72);
        color: #334155;
        font-weight: 600;
    }

    .download-card {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.26);
        background: rgba(255, 255, 255, 0.72);
        padding: 0.75rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(79, 70, 229, 0.25) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: all 160ms ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.34);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(238,242,255,0.95));
        border-right: 1px solid rgba(148, 163, 184, 0.24);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.15rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 10px 20px rgba(30, 41, 59, 0.10);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 12px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 15px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def check_backend_health(base_url: str) -> bool:
    health_url = base_url.rstrip("/") + "/health"
    try:
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def render_topbar(backend_ok: bool):
    status_color = "#10b981" if backend_ok else "#ef4444"
    status_text = "System Online" if backend_ok else "System Offline"
    st.markdown(
        f"""
        <div class="glass topbar">
            <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center;">
                <div>
                    <p class="brand-title">FEDSEG AI Segmentation Console</p>
                    <p class="brand-sub">Federated ResNet-UNet medical imaging dashboard with real-time inference insights.</p>
                    <div class="topbar-badges">
                        <span class="pill">Medical Imaging</span>
                        <span class="pill">Federated Learning</span>
                        <span class="pill">Real-time Inference</span>
                    </div>
                </div>
                <div class="status-wrap">
                    <span class="status-dot" style="background:{status_color};"></span>
                    <span>{status_text}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    tint = np.array([124, 58, 237], dtype=np.float32) if tint is None else np.array(tint, dtype=np.float32)
    foreground = mask_resized > threshold
    for c in range(3):
        overlay[:, :, c] = np.where(
            foreground,
            (1 - alpha) * overlay[:, :, c] + alpha * tint[c],
            overlay[:, :, c],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_comparison_split(original_rgb, overlay_rgb, split_pct):
    split = int(original_rgb.shape[1] * split_pct)
    compare = original_rgb.copy()
    compare[:, split:, :] = overlay_rgb[:, split:, :]
    return compare


def render_empty_state():
    st.markdown(
        """
        <div class="glass section-card">
            <div class="empty-state">
                <div class="empty-title">No study loaded</div>
                <p class="empty-sub">Upload a medical image from the left panel to run segmentation and unlock analytics.</p>
                <div class="step-row">
                    <span class="step-chip">1. Upload</span>
                    <span class="step-chip">2. Infer</span>
                    <span class="step-chip">3. Analyze</span>
                    <span class="step-chip">4. Export</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_info():
    st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Model Info</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Core inference model and deployment metadata.</div>',
        unsafe_allow_html=True,
    )
    model_data = pd.DataFrame(
        [
            {"Property": "Model Family", "Value": "ResNet-UNet"},
            {"Property": "Training Setup", "Value": "Federated Learning (FSSS)"},
            {"Property": "Input Resolution", "Value": "224 x 224 (inference preprocess)"},
            {"Property": "Output", "Value": "Single-channel segmentation mask [0,1]"},
            {"Property": "Deployment", "Value": "FastAPI + Streamlit"},
            {"Property": "Clinical Status", "Value": "Research/Demo only"},
        ]
    )
    st.dataframe(model_data, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


inject_styles()

with st.sidebar:
    st.markdown("## Control Panel")
    st.markdown("Configure endpoint, visualization, and output behavior.")
    api_base = st.text_input(
        "Backend API URL",
        value=DEFAULT_API_URL,
        help="Example: http://localhost:8000",
    )
    st.markdown("### Segmentation")
    threshold = st.slider(
        "Segmentation threshold",
        min_value=0.2,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Pixels above this value are treated as segmented region.",
    )
    colormap = st.selectbox(
        "Mask colormap",
        ["Purple (medical)", "Green (medical)", "Viridis", "Plasma", "Grayscale"],
        index=0,
    )
    overlay_strength = st.slider(
        "Overlay intensity",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Higher values make segmentation overlay stronger.",
    )
    overlay_tone = st.selectbox(
        "Overlay tone",
        ["Violet", "Cyan", "Emerald", "Amber"],
        index=0,
    )
    st.markdown("### Comparison")
    compare_split = st.slider(
        "Before/After split position",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Controls split line in the comparison view.",
    )
    show_3d = st.toggle("Enable 3D volume rendering", value=True)
    st.markdown("---")
    st.caption("Guided flow: Upload -> Process -> Review -> Export")
    st.caption("For research/demo use only.")

backend_ok = check_backend_health(api_base)
render_topbar(backend_ok)

api_url = api_base.rstrip("/") + "/predict"

st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Upload Section</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Import one PNG image to start segmentation inference.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="upload-hint">Recommended: frontal chest X-ray PNG for best visual consistency.</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("Upload image", type=["png"], accept_multiple_files=False, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    render_empty_state()
    render_model_info()
    st.stop()

file_bytes = uploaded_file.read()
np_arr = np.frombuffer(file_bytes, np.uint8)
decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if decoded is None:
    st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
    st.error("Could not decode image. Please upload a valid PNG file.")
    st.markdown("</div>", unsafe_allow_html=True)
    render_model_info()
    st.stop()

original_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
h_orig, w_orig = original_rgb.shape[:2]
filename = uploaded_file.name or "image.png"

st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Inference</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Model is processing the uploaded image and generating confidence mask.</div>',
    unsafe_allow_html=True,
)
progress = st.progress(8, text="Preparing request...")

try:
    progress.progress(30, text="Sending image to inference service...")
    with st.spinner("Running segmentation model..."):
        response = requests.post(api_url, files={"file": file_bytes}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    progress.progress(100, text="Inference complete")
except requests.exceptions.RequestException as exc:
    st.error(f"API error: {exc}")
    if hasattr(exc, "response") and exc.response is not None:
        try:
            st.json(exc.response.json())
        except Exception:
            st.code(exc.response.text)
    st.markdown("</div>", unsafe_allow_html=True)
    render_model_info()
    st.stop()

st.success("Segmentation completed successfully.")
st.markdown("</div>", unsafe_allow_html=True)

mask = np.array(payload["mask"], dtype=np.float32)
mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
binary = (mask_resized > threshold).astype(np.float32)

area_pct = 100.0 * binary.sum() / (binary.size or 1)
mean_conf = float(np.mean(mask_resized))
std_conf = float(np.std(mask_resized))
inf_ms = payload.get("inference_time_ms")
foreground_pixels = int(binary.sum())
total_pixels = int(binary.size)

mask_rgb = mask_to_rgb(mask_resized, colormap)
tone_map = {
    "Violet": [124, 58, 237],
    "Cyan": [6, 182, 212],
    "Emerald": [16, 185, 129],
    "Amber": [245, 158, 11],
}
overlay = make_overlay(
    original_rgb,
    mask_resized,
    threshold,
    alpha=overlay_strength,
    tint=tone_map[overlay_tone],
)
compare_view = make_comparison_split(original_rgb, overlay, compare_split)

attributes = {
    "Filename": filename,
    "Image width (px)": payload.get("image_width", w_orig),
    "Image height (px)": payload.get("image_height", h_orig),
    "Area segmented (%)": round(area_pct, 2),
    "Mean confidence": round(mean_conf, 4),
    "Std confidence": round(std_conf, 4),
    "Mask min": round(payload.get("mask_min", float(np.min(mask))), 4),
    "Mask max": round(payload.get("mask_max", float(np.max(mask))), 4),
    "Threshold": threshold,
    "Inference time (ms)": inf_ms,
    "Model": "FEDSEG ResNet-UNet (FSSS)",
}

st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Results Section</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Key outputs and confidence indicators for current image.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Area Segmented", f"{area_pct:.1f}%")
with m2:
    render_metric_card("Mean Confidence", f"{mean_conf:.3f}")
with m3:
    render_metric_card("Inference Time", f"{inf_ms} ms" if inf_ms is not None else "N/A")
with m4:
    render_metric_card("Threshold", f"{threshold:.2f}")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        f"""
        <div class="mini-stat">
            <p class="mini-stat-label">Segmented Pixels</p>
            <p class="mini-stat-value">{foreground_pixels:,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        f"""
        <div class="mini-stat">
            <p class="mini-stat-label">Total Pixels</p>
            <p class="mini-stat-value">{total_pixels:,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        f"""
        <div class="mini-stat">
            <p class="mini-stat-label">Input Resolution</p>
            <p class="mini-stat-value">{w_orig} x {h_orig}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_viz, tab_analytics, tab_table, tab_attrs = st.tabs(
    ["Visualization", "Analytics", "Attributes Table", "Attribute Details"]
)

with tab_viz:
    st.markdown("### Image Comparison")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Original")
        st.image(original_rgb, use_container_width=True)
    with c2:
        st.caption("Predicted Mask")
        st.image(mask_rgb, use_container_width=True)
    with c3:
        st.caption("Overlay")
        st.image(overlay, use_container_width=True)

    st.markdown("### Before/After Split View")
    st.caption("Left side: original image | Right side: overlay result")
    st.image(compare_view, use_container_width=True)

    st.markdown("### Export")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="download-card">', unsafe_allow_html=True)
        mask_uint8 = (np.clip(mask_resized, 0, 1) * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
        _, mask_buf = cv2.imencode(".png", mask_bgr)
        st.download_button(
            "Download Mask PNG",
            data=mask_buf.tobytes(),
            file_name="mask.png",
            mime="image/png",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="download-card">', unsafe_allow_html=True)
        export_df = pd.DataFrame([{"Attribute": k, "Value": v} for k, v in attributes.items()])
        st.download_button(
            "Download Attributes CSV",
            data=export_df.to_csv(index=False),
            file_name="attributes.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="download-card">', unsafe_allow_html=True)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, overlay_buf = cv2.imencode(".png", overlay_bgr)
        st.download_button(
            "Download Overlay PNG",
            data=overlay_buf.tobytes(),
            file_name="overlay.png",
            mime="image/png",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab_analytics:
    st.markdown("### Confidence Analytics")
    g1, g2 = st.columns(2)

    with g1:
        hist_vals, bin_edges = np.histogram(mask_resized.flatten(), bins=50, range=(0, 1))
        fig_hist = go.Figure(
            data=[
                go.Bar(
                    x=(bin_edges[:-1] + bin_edges[1:]) / 2,
                    y=hist_vals,
                    marker_color="#4f46e5",
                    opacity=0.88,
                )
            ]
        )
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#06b6d4",
            annotation_text="Threshold",
            annotation_position="top",
        )
        fig_hist.update_layout(
            title="Mask Confidence Distribution",
            xaxis_title="Confidence",
            yaxis_title="Pixel Count",
            template="plotly_white",
            margin=dict(t=48, b=32, l=30, r=18),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.68)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with g2:
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        counts, _ = np.histogram(mask_resized.flatten(), bins=bins)
        labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
        fig_band = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=counts,
                    marker_color=["#c7d2fe", "#a5b4fc", "#818cf8", "#4f46e5"],
                    opacity=0.92,
                )
            ]
        )
        fig_band.update_layout(
            title="Confidence Bands",
            xaxis_title="Range",
            yaxis_title="Pixels",
            template="plotly_white",
            margin=dict(t=48, b=32, l=30, r=18),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.68)",
        )
        st.plotly_chart(fig_band, use_container_width=True)

    if show_3d:
        st.markdown("### 3D Volume Projection")
        vol = np.repeat(mask_resized[np.newaxis, ...], 20, axis=0)
        zz, yy, xx = np.meshgrid(
            np.arange(vol.shape[0]),
            np.arange(vol.shape[1]),
            np.arange(vol.shape[2]),
            indexing="ij",
        )
        fig3d = go.Figure(
            data=go.Volume(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=vol.flatten(),
                opacity=0.12,
                surface_count=18,
                colorscale=[
                    [0.0, "rgba(79,70,229,0.00)"],
                    [0.5, "rgba(79,70,229,0.45)"],
                    [1.0, "rgba(6,182,212,0.95)"],
                ],
            )
        )
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=10),
            height=430,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                bgcolor="rgb(248, 250, 252)",
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("3D volume rendering is disabled in the left panel.")

with tab_table:
    table_df = pd.DataFrame([{"Attribute": k, "Value": v} for k, v in attributes.items()])
    st.dataframe(table_df, use_container_width=True, hide_index=True)

with tab_attrs:
    for k, v in attributes.items():
        st.markdown(f"**{k}**  \n`{v}`")

st.markdown("</div>", unsafe_allow_html=True)
render_model_info()
