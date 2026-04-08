import cv2
import numpy as np
import requests
import streamlit as st

try:
    from frontend.api_client import check_backend_health, decode_mask_png_base64, request_prediction
    from frontend.config import DEFAULT_API_URL
    from frontend.image_utils import decode_uploaded_image, make_comparison_split, make_overlay, mask_to_rgb
    from frontend.styles import inject_styles
    from frontend.ui import build_attributes, render_empty_state, render_model_info, render_results, render_topbar
except ModuleNotFoundError:
    # Fallback for environments where Streamlit runs this file without repo root on sys.path.
    from api_client import check_backend_health, decode_mask_png_base64, request_prediction
    from config import DEFAULT_API_URL
    from image_utils import decode_uploaded_image, make_comparison_split, make_overlay, mask_to_rgb
    from styles import inject_styles
    from ui import build_attributes, render_empty_state, render_model_info, render_results, render_topbar

st.set_page_config(
    page_title="FEDSEG AI Dashboard",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar():
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
    return {
        "api_base": api_base,
        "threshold": threshold,
        "colormap": colormap,
        "overlay_strength": overlay_strength,
        "overlay_tone": overlay_tone,
        "compare_split": compare_split,
        "show_3d": show_3d,
    }


def render_upload():
    st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Upload Section</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Import one PNG or JPEG image to start segmentation inference.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="upload-hint">Recommended: frontal chest X-ray PNG or JPEG for best visual consistency.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file


def main():
    inject_styles()
    controls = render_sidebar()

    backend_ok = check_backend_health(controls["api_base"])
    render_topbar(backend_ok)

    uploaded_file = render_upload()
    if uploaded_file is None:
        render_empty_state()
        render_model_info()
        st.stop()

    file_bytes = uploaded_file.read()
    decoded = decode_uploaded_image(file_bytes)
    if decoded is None:
        st.markdown('<div class="glass section-card">', unsafe_allow_html=True)
        st.error("Could not decode image. Please upload a valid PNG or JPEG file.")
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
            payload = request_prediction(controls["api_base"].rstrip("/") + "/predict", file_bytes)
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

    try:
        mask = decode_mask_png_base64(payload["mask_png_base64"])
    except (KeyError, ValueError) as exc:
        st.error(f"Invalid model response: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
        render_model_info()
        st.stop()

    st.success("Segmentation completed successfully.")
    st.markdown("</div>", unsafe_allow_html=True)

    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    area_pct = 100.0 * (mask_resized > controls["threshold"]).sum() / (mask_resized.size or 1)
    mean_conf = float(np.mean(mask_resized))
    std_conf = float(np.std(mask_resized))

    mask_rgb = mask_to_rgb(mask_resized, controls["colormap"])
    tone_map = {
        "Violet": [79, 70, 229],
        "Cyan": [14, 165, 233],
        "Emerald": [16, 185, 129],
        "Amber": [245, 158, 11],
    }
    overlay = make_overlay(
        original_rgb,
        mask_resized,
        controls["threshold"],
        alpha=controls["overlay_strength"],
        tint=tone_map[controls["overlay_tone"]],
    )
    compare_view = make_comparison_split(original_rgb, overlay, controls["compare_split"])
    attributes = build_attributes(
        filename,
        payload,
        w_orig,
        h_orig,
        area_pct,
        mean_conf,
        std_conf,
        controls["threshold"],
    )
    render_results(
        original_rgb,
        mask_rgb,
        overlay,
        compare_view,
        mask_resized,
        attributes,
        controls["threshold"],
        controls["show_3d"],
    )
    render_model_info()


if __name__ == "__main__":
    main()
