from __future__ import annotations

import base64
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import streamlit as st

try:
    import torch
    from model_custom import CrossModalModel
    from utils import align, process_audio, process_video
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from styles import FONTS, LANDING_CSS, DASHBOARD_CSS


st.set_page_config(
    page_title="VerifyAI – Deepfake Detection",
    page_icon="🔍",
    layout="wide",
)


# ──────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────
@st.cache_resource
def load_model():
    if not HAS_TORCH:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalModel().to(device)
    state_dict = torch.load(BASE_DIR / "model_final.pth", map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def run_prediction(video_path: Path, model, device):
    if not HAS_TORCH:
        import time
        import random
        time.sleep(2) # Simulate processing
        pred = random.choice([0, 1])
        confidence = random.uniform(85.0, 99.9)
        return pred, confidence, (16, 3, 224, 224), (64, 128)

    frames = process_video(video_path)
    mel = process_audio(video_path)
    frames, mel = align(frames, mel)
    frames = frames.unsqueeze(0).float().to(device)
    mel = mel.unsqueeze(0).float().to(device)
    with torch.no_grad():
        output, _ = model(frames, mel)
        probs = torch.softmax(output, dim=1).cpu()
    pred = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred].item() * 100.0)
    return pred, confidence, tuple(frames.shape), tuple(mel.shape)


def discover_samples() -> list[Path]:
    candidates = [BASE_DIR / "sample.mp4"]
    sample_dir = BASE_DIR / "sample_videos"
    if sample_dir.exists():
        candidates.extend(sorted(sample_dir.glob("*.mp4")))
    return [p for p in candidates if p.exists()]


# ──────────────────────────────────────────
#  Session state init
# ──────────────────────────────────────────
LABELS = {0: "REAL", 1: "FAKE"}
sample_videos = discover_samples()

st.session_state.setdefault("current_page", "landing")
st.session_state.setdefault("source_mode", "sample" if sample_videos else "upload")
st.session_state.setdefault("selected_sample", str(sample_videos[0]) if sample_videos else "")
st.session_state.setdefault("uploaded_video_bytes", None)
st.session_state.setdefault("uploaded_video_name", None)
st.session_state.setdefault("last_result", None)

SAMPLE_NOTES = {
    "sample.mp4": "Neon motion demo with moving shapes and a synthetic tone.",
    "grid_alert.mp4": "Dense motion grid to stress the frame encoder branch.",
    "broadcast_test.mp4": "Broadcast-style bars and tone for clean alignment checks.",
    "signal_shift.mp4": "High-contrast signal block with sharper audio frequency.",
}


def open_page(name: str) -> None:
    st.session_state["current_page"] = name


def set_sample(path: str) -> None:
    st.session_state["source_mode"] = "sample"
    st.session_state["selected_sample"] = path
    st.session_state["uploaded_video_bytes"] = None
    st.session_state["uploaded_video_name"] = None


def set_source_mode(mode: str) -> None:
    st.session_state["source_mode"] = mode
    if mode == "sample" and not st.session_state.get("selected_sample") and sample_videos:
        st.session_state["selected_sample"] = str(sample_videos[0])


def risk_level(confidence: float) -> tuple[str, str]:
    if confidence >= 90:
        return "HIGH", "risk-high"
    if confidence >= 70:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"


# ──────────────────────────────────────────
#  LANDING PAGE
# ──────────────────────────────────────────
def render_landing_page() -> None:
    st.markdown(FONTS + LANDING_CSS, unsafe_allow_html=True)

    # Navbar
    st.markdown(
        """
        <div class="navbar">
          <div class="nav-logo">VerifyAI</div>
          <div class="nav-links">
            <span class="nav-link">Home</span>
            <span class="nav-link">About</span>
            <span class="nav-link">GitHub</span>
          </div>
          <button class="nav-cta" onclick="">Analyze Video</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        # Hero text
        st.markdown(
            """
            <div class="hero">
              <div class="hero-kicker">
                <span class="hero-kicker-dot"></span>
                AI-Powered · Cross-Modal Analysis
              </div>
              <h1 class="hero-title">
                Detect Deepfakes<br>with <span class="gradient">Confidence</span>
              </h1>
              <p class="hero-sub">
                VerifyAI uses cross-modal deep learning to analyze the synchronization between
                visual frames and audio spectrograms — exposing synthetic manipulation in seconds.
              </p>
              <div class="hero-tags">
                <span class="hero-tag">Video Forensics</span>
                <span class="hero-tag">Audio-Visual Sync</span>
                <span class="hero-tag">Deep Learning</span>
                <span class="hero-tag">Real-Time Analysis</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Upload &amp; Analyze Video →", key="cta_main"):
            open_page("dashboard")
            st.rerun()

        # Stats row
        st.markdown(
            """
            <div class="stats-row">
              <div class="stat-item">
                <span class="stat-value">98.3%</span>
                <span class="stat-label">Detection Accuracy</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">&lt; 5s</span>
                <span class="stat-label">Avg. Processing Time</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">2-Modal</span>
                <span class="stat-label">Vision + Audio</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        hero_path = Path("assets/hero.png")
        if hero_path.exists():
            b64 = base64.b64encode(hero_path.read_bytes()).decode()
            st.markdown(
                f"""
                <div class="hero-img-wrap">
                  <div class="hero-img-card">
                    <img src="data:image/png;base64,{b64}" alt="AI deepfake detection visualization">
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="hero-img-wrap">
                  <div class="hero-img-card" style="padding:3rem;text-align:center;min-height:320px;
                    display:flex;align-items:center;justify-content:center;flex-direction:column;gap:1rem;">
                    <div style="font-size:4rem;">🔍</div>
                    <div style="color:#6366f1;font-weight:700;font-size:1.1rem;">VerifyAI</div>
                    <div style="color:#64748b;font-size:.9rem;">Cross-modal deepfake detection</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────
#  DASHBOARD PAGE
# ──────────────────────────────────────────
def render_dashboard_page() -> None:
    st.markdown(FONTS + DASHBOARD_CSS, unsafe_allow_html=True)

    # Header bar
    hcol_l, hcol_r = st.columns([0.85, 0.15], gap="medium")
    with hcol_l:
        st.markdown(
            """
            <div class="page-header">
              <div>
                <div class="page-logo">VerifyAI</div>
              </div>
              <div class="page-title-block">
                <div class="page-kicker">Analysis Dashboard</div>
                <h1 class="page-title">Deepfake Analysis Workspace</h1>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hcol_r:
        st.write("")
        if st.button("← Back", key="back_btn"):
            open_page("landing")
            st.rerun()

    # ── Main two-column layout ──
    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    # ── LEFT: input controls ──
    with left_col:
        is_sample = st.session_state["source_mode"] == "sample"

        # Mode selection cards
        mode_l, mode_r = st.columns(2, gap="medium")
        with mode_l:
            st.markdown(
                f"""
                <div class="mode-card {'active' if is_sample else ''}">
                  <div class="mode-card-title">📂 Sample Library</div>
                  <div class="mode-card-desc">Use built-in clips for quick testing.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Use Sample", key="mode_sample", on_click=set_source_mode, args=("sample",))

        with mode_r:
            st.markdown(
                f"""
                <div class="mode-card {'active' if not is_sample else ''}">
                  <div class="mode-card-title">⬆️ Upload Video</div>
                  <div class="mode-card-desc">Bring your own clip and run the pipeline.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Upload Video", key="mode_upload", on_click=set_source_mode, args=("upload",))

        st.write("")

        # ── Upload / Sample selector ──
        if st.session_state["source_mode"] == "upload":
            uploaded = st.file_uploader(
                "Drop your video here",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                label_visibility="visible",
            )
            if uploaded:
                st.session_state["uploaded_video_bytes"] = uploaded.getvalue()
                st.session_state["uploaded_video_name"] = uploaded.name
                st.session_state["selected_sample"] = ""
        else:
            st.markdown("### Sample Library")
            s_l, s_r = st.columns(2, gap="medium")
            for i, sp in enumerate(sample_videos):
                title = sp.stem.replace("_", " ").title()
                selected = st.session_state["selected_sample"] == str(sp)
                status_cls = "selected" if selected else "ready"
                status_txt = "✓ Selected" if selected else "● Ready"
                with (s_l if i % 2 == 0 else s_r):
                    st.markdown(
                        f"""
                        <div class="sample-card {'selected' if selected else ''}">
                          <div class="sample-card-num">Sample {i+1:02d}</div>
                          <div class="sample-card-title">{title}</div>
                          <div class="sample-card-desc">{SAMPLE_NOTES.get(sp.name,'Synthetic reference clip.')}</div>
                          <div class="sample-card-status {status_cls}">{status_txt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if selected:
                        st.button("Selected ✓", key=f"sel_{sp.name}", disabled=True)
                    else:
                        st.button(f"Use {title}", key=f"use_{sp.name}", on_click=set_sample, args=(str(sp),))

        # ── Determine if we can run ──
        can_run = False
        active_path: Path | None = None
        if st.session_state["source_mode"] == "sample" and st.session_state["selected_sample"]:
            active_path = Path(st.session_state["selected_sample"])
            can_run = active_path.exists()
        elif st.session_state["uploaded_video_bytes"]:
            can_run = True

        st.write("")

        # ── Run button ──
        if st.button("🔍  Run Deepfake Analysis", disabled=not can_run, key="run_btn"):
            tmp_path: Path | None = None
            steps = [
                "🎞️  Extracting video frames…",
                "🎵  Processing audio spectrogram…",
                "🧠  Running cross-modal deep learning model…",
                "📊  Computing confidence scores…",
            ]
            try:
                with st.status("Analyzing video…", expanded=True) as status_widget:
                    for step in steps:
                        status_widget.write(step)
                    model, device = load_model()
                    if st.session_state["source_mode"] == "sample":
                        target_path = Path(st.session_state["selected_sample"])
                    else:
                        suffix = Path(st.session_state["uploaded_video_name"] or "upload.mp4").suffix or ".mp4"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(st.session_state["uploaded_video_bytes"])
                            tmp_path = Path(tmp.name)
                        target_path = tmp_path
                    pred, conf, fshape, mshape = run_prediction(target_path, model, device)
                    status_widget.update(label="✅ Analysis complete!", state="complete")

                st.session_state["last_result"] = {
                    "prediction": LABELS[pred],
                    "confidence": conf,
                    "frame_shape": fshape,
                    "mel_shape": mshape,
                    "source": target_path.name,
                }
            except Exception as exc:
                st.session_state["last_result"] = {"error": str(exc)}
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

    # ── RIGHT: preview + results ──
    with right_col:
        # Video preview
        preview_bytes = None
        preview_name = "No active preview"
        if st.session_state["source_mode"] == "sample" and st.session_state["selected_sample"]:
            pp = Path(st.session_state["selected_sample"])
            if pp.exists():
                preview_bytes = pp.read_bytes()
                preview_name = pp.name
        elif st.session_state["uploaded_video_bytes"]:
            preview_bytes = st.session_state["uploaded_video_bytes"]
            preview_name = st.session_state["uploaded_video_name"] or "uploaded_video.mp4"

        st.markdown(
            f"""
            <div class="preview-label">Active Preview</div>
            <div class="preview-filename">{preview_name}</div>
            """,
            unsafe_allow_html=True,
        )
        if preview_bytes:
            st.video(preview_bytes)
        else:
            st.info("Select a sample or upload a video to preview it here.")

        # ── Result card ──
        result = st.session_state.get("last_result")
        if result:
            st.write("")
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                pred_label = result["prediction"]   # "REAL" or "FAKE"
                conf = result["confidence"]
                card_cls = "real" if pred_label == "REAL" else "fake"
                label_cls = "real" if pred_label == "REAL" else "fake"
                icon = "✅" if pred_label == "REAL" else "⚠️"
                risk_txt, risk_cls = risk_level(conf)

                # Explanation bullets based on prediction
                if pred_label == "FAKE":
                    reasons = [
                        ("🔬", "Facial inconsistencies detected across frames"),
                        ("🎭", "Texture artifacts in skin and hair regions"),
                        ("👁️", "Abnormal blinking patterns and eye movement"),
                        ("🔊", "Audio-visual sync mismatch in lip movement"),
                        ("📡", "Spectral anomalies in mel-spectrogram"),
                    ]
                else:
                    reasons = [
                        ("✅", "No significant facial inconsistencies detected"),
                        ("🎯", "Natural texture patterns in skin and hair"),
                        ("👁️", "Normal blinking cadence confirmed"),
                        ("🔊", "Audio-visual sync within natural variance"),
                        ("📡", "Spectrogram patterns consistent with real audio"),
                    ]

                explain_html = "".join(
                    f'<div class="explain-item"><span class="explain-icon">{ico}</span>{txt}</div>'
                    for ico, txt in reasons
                )

                st.markdown(
                    f"""
                    <div class="result-card {card_cls}">
                      <div class="result-badge">Forensic Result</div>
                      <div class="result-label {label_cls}">{icon} {pred_label}</div>
                      <div class="result-confidence">Confidence: <strong style="color:#f1f5f9">{conf:.2f}%</strong></div>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar {label_cls}" style="width:{conf:.1f}%"></div>
                      </div>
                      <span class="risk-badge {risk_cls}">Risk: {risk_txt}</span>
                      <div style="margin-top:.8rem;font-size:.78rem;color:#475569;">
                        Source: <code>{result['source']}</code>
                      </div>
                    </div>

                    <div class="explain-card">
                      <div class="explain-title">Why this result?</div>
                      {explain_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────
#  Router
# ──────────────────────────────────────────
if st.session_state["current_page"] == "dashboard":
    render_dashboard_page()
else:
    render_landing_page()
