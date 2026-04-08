import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
            {"Property": "Transport", "Value": "Compressed PNG mask payload"},
            {"Property": "Deployment", "Value": "FastAPI + Streamlit"},
            {"Property": "Clinical Status", "Value": "Research/Demo only"},
        ]
    )
    st.dataframe(model_data, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def build_attributes(filename, payload, width, height, area_pct, mean_conf, std_conf, threshold):
    return {
        "Filename": filename,
        "Image width (px)": payload.get("image_width", width),
        "Image height (px)": payload.get("image_height", height),
        "Area segmented (%)": round(area_pct, 2),
        "Mean confidence": round(mean_conf, 4),
        "Std confidence": round(std_conf, 4),
        "Mask min": round(payload.get("mask_min", 0.0), 4),
        "Mask max": round(payload.get("mask_max", 1.0), 4),
        "Threshold": threshold,
        "Inference time (ms)": payload.get("inference_time_ms"),
        "Mask transport": payload.get("mask_encoding", "png_base64"),
        "Model": "FEDSEG ResNet-UNet (FSSS)",
    }


def render_results(original_rgb, mask_rgb, overlay, compare_view, mask_resized, attributes, threshold, show_3d):
    area_pct = attributes["Area segmented (%)"]
    mean_conf = attributes["Mean confidence"]
    inf_ms = attributes["Inference time (ms)"]
    foreground_pixels = int((mask_resized > threshold).sum())
    total_pixels = int(mask_resized.size)
    height, width = original_rgb.shape[:2]

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
                <p class="mini-stat-value">{width} x {height}</p>
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
            _, mask_buf = cv2.imencode(".png", mask_uint8)
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
                        marker_color="#1e40af",
                        opacity=0.88,
                    )
                ]
            )
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="#0ea5e9",
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
                        marker_color=["#dbeafe", "#93c5fd", "#3b82f6", "#1e40af"],
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
                        [0.0, "rgba(30,64,175,0.00)"],
                        [0.5, "rgba(30,64,175,0.45)"],
                        [1.0, "rgba(14,165,233,0.95)"],
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
        # Force string values to avoid Arrow serialization issues with mixed object types.
        table_df = pd.DataFrame([{"Attribute": str(k), "Value": str(v)} for k, v in attributes.items()])
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    with tab_attrs:
        for key, value in attributes.items():
            st.markdown(f"**{key}**  \n`{value}`")

    st.markdown("</div>", unsafe_allow_html=True)
