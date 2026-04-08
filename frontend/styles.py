import streamlit as st


def inject_styles():
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-0: #f5f8fc;
        --bg-1: #edf3fb;
        --bg-2: #e8f0f8;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --card-bg: rgba(255, 255, 255, 0.72);
        --card-border: rgba(148, 163, 184, 0.22);
        --primary: #1e40af;
        --primary-soft: #2563eb;
        --accent: #0ea5e9;
        --success: #10b981;
        --danger: #ef4444;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background:
            radial-gradient(860px 370px at 8% 0%, rgba(30, 64, 175, 0.14), transparent 60%),
            radial-gradient(760px 320px at 92% 0%, rgba(14, 165, 233, 0.12), transparent 60%),
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
        box-shadow: 0 6px 18px rgba(30, 64, 175, 0.09);
        transition: transform 150ms ease, box-shadow 150ms ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(30, 64, 175, 0.16);
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
        border: 1px dashed rgba(37, 99, 235, 0.32);
        background: rgba(14, 165, 233, 0.08);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        color: #0f4c81;
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
        border: 1px solid rgba(30, 64, 175, 0.24) !important;
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: all 160ms ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(30, 64, 175, 0.30);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(237,243,251,0.96));
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
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        color: white;
    }
</style>
        """,
        unsafe_allow_html=True,
    )
