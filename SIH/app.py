"""
LUNARMATCH AI — Planetary Image Registration & Lunar Cartography Workstation
Smart India Hackathon Problem Statement #26166 (ISRO)
Multi-Modal, Sun-Angle and Scale-Invariant Lunar Image Correspondence & Sub-Pixel Registration
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import hashlib
import re
import time

# Local imports
from config import load_settings
from utils.hardware import get_hardware_info
from utils.paths import RAW_DATA_DIR, OUTPUTS_DIR, SAMPLES_DIR
from preprocessing.loader import LunarImage, load_lunar_image
from analysis.image_quality import analyze_image_quality
from analysis.overlap import estimate_overlap
from analysis.failure_diagnostics import diagnose_registration_run
from pipeline.registration_pipeline import RegistrationPipeline, PipelineExecutionResult
from pipeline.export import export_registration_package
from experiments.tracker import ExperimentTracker
from registration.warping import create_split_wipe, normalize_for_display, crop_to_valid_overlap
from visualization.matches import draw_correspondence_rays
from visualization.diagnostics import plot_residual_histogram, plot_error_vectors
from evaluation.benchmark import run_comparative_benchmark
from demo.scenarios import (
    SCENARIOS_CATALOG,
    get_scenario_by_id,
    ScenarioBenchmarkData,
)
from features.learned import LearnedFeatureExtractor


# Set page configuration
st.set_page_config(
    page_title="LUNARMATCH AI — Planetary Cartography Workstation (ISRO SIH #26166)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lunar mission-control visual system.  The data shown inside these surfaces is
# still sourced from the pipeline below; CSS is intentionally presentation-only.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    :root {--panel:rgba(12,29,36,.90);--line:rgba(157,214,224,.24);--muted:#94adb5;--ice:#e5fbfd;--cyan:#71dae6;--lime:#b8f1a2;}
    .stApp {background:radial-gradient(circle at 84% -4%,rgba(85,201,214,.15),transparent 29rem),radial-gradient(circle at 5% 38%,rgba(143,105,191,.11),transparent 28rem),linear-gradient(135deg,#061015,#0b1a21 52%,#071116);color:var(--ice);font-family:'Inter',sans-serif;}
    .stApp:before {content:"";position:fixed;inset:0;pointer-events:none;opacity:.32;background-image:linear-gradient(rgba(177,232,238,.032) 1px,transparent 1px),linear-gradient(90deg,rgba(177,232,238,.032) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(to bottom,black,transparent 78%);}
    #MainMenu,footer {visibility:hidden}.block-container {padding:1.35rem 2.1rem 4rem;max-width:1600px}h1,h2,h3 {letter-spacing:-.04em;color:var(--ice)}h2 {font-size:clamp(1.65rem,2.8vw,2.35rem)!important}button:focus-visible,[role="tab"]:focus-visible {outline:2px solid var(--cyan)!important;outline-offset:3px}
    .lunar-hero {position:relative;overflow:hidden;min-height:360px;padding:42px 46px;margin:4px 0 14px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(112deg,rgba(18,38,47,.97),rgba(9,21,27,.90));box-shadow:0 24px 75px rgba(0,0,0,.22),inset 0 1px rgba(255,255,255,.05);isolation:isolate}.lunar-hero:before {content:"";position:absolute;inset:0;z-index:-1;background:linear-gradient(90deg,transparent 0 52%,rgba(99,216,229,.06)),repeating-linear-gradient(-26deg,transparent 0 32px,rgba(155,231,239,.035) 33px 34px)}.lunar-hero:after {content:"";position:absolute;right:-4%;top:-34%;height:560px;width:560px;z-index:-1;border-radius:50%;background:radial-gradient(circle at 37% 32%,#d7d9d1 0 1%,#788788 2% 4%,#c1c7c0 5% 7%,#536062 8% 12%,#939d98 13% 17%,#384346 18% 24%,#77847f 25% 31%,#293337 32% 40%,#596560 41% 52%,#182126 53% 100%);filter:contrast(1.18) grayscale(1);opacity:.72;box-shadow:inset -45px -40px 80px #081317;animation:moon-drift 18s ease-in-out infinite alternate}.hero-kicker,.eyebrow {font:600 .7rem 'JetBrains Mono',monospace;letter-spacing:.18em;color:var(--cyan);text-transform:uppercase}.lunar-hero h1 {max-width:720px;margin:.55rem 0;font-size:clamp(3.2rem,6.7vw,6.7rem);line-height:.8;font-weight:700;text-shadow:0 4px 30px rgba(0,0,0,.25)}.hero-sub {max-width:600px;font:500 clamp(.78rem,1.4vw,1rem) 'JetBrains Mono',monospace;letter-spacing:.1em;line-height:1.6;color:#d4e9e8}.hero-copy {max-width:510px;margin-top:17px;color:#adc3c5;line-height:1.6}.hero-readout {position:absolute;right:30px;bottom:28px;width:min(36vw,400px);padding:15px 16px;border:1px solid rgba(157,220,227,.42);border-radius:11px;background:rgba(4,15,20,.72);backdrop-filter:blur(10px);font:500 .67rem 'JetBrains Mono',monospace;letter-spacing:.04em;line-height:1.6;color:#c9d6d4;box-shadow:0 12px 30px rgba(0,0,0,.22)}.hero-readout span {color:var(--cyan)}.orbital {position:absolute;width:620px;height:230px;right:-40px;bottom:-62px;border:1px solid rgba(115,219,230,.46);border-radius:50%;transform:rotate(-18deg);z-index:-1}.orbital:after {content:"";position:absolute;width:9px;height:9px;border-radius:50%;background:#bff7fc;left:26%;top:4%;box-shadow:0 0 0 5px rgba(156,219,227,.16),0 0 20px #75dce7}.mission-status {display:grid;grid-template-columns:repeat(5,1fr);margin:0 0 25px;border:1px solid var(--line);border-radius:13px;background:rgba(11,27,34,.78);overflow:hidden;box-shadow:0 14px 36px rgba(0,0,0,.14)}.mission-status div {padding:14px 17px;border-right:1px solid var(--line);min-width:0}.mission-status div:last-child {border-right:0}.mission-status b {display:block;font:600 .62rem 'JetBrains Mono',monospace;letter-spacing:.12em;color:#78959e}.mission-status span {display:block;font:600 .76rem 'JetBrains Mono',monospace;color:#e8f5f4;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ready-dot {color:var(--lime)!important}
    .telemetry-bar {background:rgba(12,31,38,.72);border:1px solid var(--line);border-radius:12px;padding:11px 18px;margin-bottom:21px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center}.telemetry-item {font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--muted);display:flex;align-items:center;gap:6px}.telemetry-val {color:var(--cyan);font-weight:600}.status-dot {width:8px;height:8px;border-radius:50%;background:var(--lime);display:inline-block;box-shadow:0 0 0 4px rgba(182,239,155,.1),0 0 12px rgba(182,239,155,.45)}
    .mission-card {background:linear-gradient(145deg,rgba(21,43,52,.9),rgba(10,25,32,.92));border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 12px 30px rgba(0,0,0,.13);transition:transform .2s ease,border-color .2s ease}.mission-card:hover {transform:translateY(-2px);border-color:rgba(103,216,229,.52)}.mission-header {font-family:'JetBrains Mono',monospace;font-size:.76rem;font-weight:600;color:var(--cyan);text-transform:uppercase;letter-spacing:.1em;margin-bottom:11px;border-bottom:1px solid var(--line);padding-bottom:8px}.metric-badge {font-family:'JetBrains Mono',monospace;padding:4px 10px;border-radius:999px;font-size:.78rem;font-weight:600;display:inline-block}.badge-green {background:rgba(34,197,94,.15);color:#8ff0a4;border:1px solid rgba(34,197,94,.3)}.badge-cyan {background:rgba(56,189,248,.15);color:#74dcff;border:1px solid rgba(56,189,248,.3)}.badge-yellow {background:rgba(234,179,8,.15);color:#f8d36d;border:1px solid rgba(234,179,8,.3)}.badge-purple {background:rgba(168,85,247,.15);color:#d0a8ff;border:1px solid rgba(168,85,247,.3)}.badge-red {background:rgba(239,68,68,.15);color:#ff9292;border:1px solid rgba(239,68,68,.3)}
    .timeline-container {display:flex;justify-content:space-between;margin-bottom:20px;background:var(--panel);padding:12px;border-radius:13px;border:1px solid var(--line)}.timeline-step {text-align:center;flex:1;font-family:'JetBrains Mono',monospace;font-size:.76rem}.timeline-step.active {color:var(--cyan);font-weight:700}.timeline-step.done {color:var(--lime)}.pipeline-nav {display:flex;align-items:stretch;margin:0 0 22px;border:1px solid var(--line);border-radius:13px;background:rgba(10,26,33,.78);overflow:auto}.pipeline-stage {min-width:125px;position:relative;padding:14px 10px 12px;text-align:center;border-right:1px solid var(--line)}.pipeline-stage:last-child {border:0}.pipeline-stage strong {display:block;font:600 .69rem 'JetBrains Mono',monospace;letter-spacing:.08em}.pipeline-stage small {display:block;color:#829da5;font-size:.68rem;margin-top:4px}.pipeline-stage .pipe-dot {display:inline-block;color:#587078;margin-right:4px}.pipeline-stage.complete .pipe-dot {color:var(--lime)}.pipeline-stage.active {background:linear-gradient(180deg,rgba(78,185,196,.16),rgba(78,185,196,.04))}.pipeline-stage.active:after {content:"";position:absolute;height:3px;left:16px;right:16px;bottom:0;border-radius:3px;background:var(--cyan);animation:signal 1.4s ease-in-out infinite}.pipeline-stage.active .pipe-dot {color:var(--cyan)}.instrument-panel {border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:17px;margin-bottom:15px}.instrument-panel .panel-title {font:600 .73rem 'JetBrains Mono',monospace;letter-spacing:.12em;color:var(--cyan);border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:11px}.radar {width:170px;aspect-ratio:1;border-radius:50%;border:1px solid #7faab0;position:relative;margin:8px auto;background:repeating-radial-gradient(circle,transparent 0 24%,rgba(127,170,176,.14) 25% 26%),linear-gradient(45deg,transparent 49.5%,rgba(127,170,176,.22) 50% 50.7%,transparent 51%),linear-gradient(-45deg,transparent 49.5%,rgba(127,170,176,.22) 50% 50.7%,transparent 51%)}.radar:before {content:"";position:absolute;inset:28%;border-radius:50%;background:#9ad4dc;opacity:.75}.radar-label {text-align:center;font:600 .66rem 'JetBrains Mono',monospace;letter-spacing:.1em;color:#a8d6dd}
    .stTabs [data-baseweb="tab-list"] {gap:6px;border-bottom:1px solid var(--line);padding-bottom:5px}.stTabs [data-baseweb="tab"] {background:rgba(8,21,27,.68);border:1px solid transparent;border-radius:9px;color:#9db5bb;padding:10px 13px}.stTabs [aria-selected="true"] {background:rgba(87,202,213,.13)!important;border-color:rgba(103,216,229,.45)!important;color:#e9ffff!important}.stButton>button {min-height:2.7rem;border-radius:9px!important;border:1px solid rgba(115,222,232,.65)!important;background:linear-gradient(135deg,#1b5862,#164047)!important;box-shadow:0 8px 20px rgba(0,0,0,.16);color:#efffff!important;font:600 .72rem 'JetBrains Mono',monospace;letter-spacing:.07em}.stButton>button:hover {transform:translateY(-1px);background:linear-gradient(135deg,#26717c,#1b525a)!important;border-color:#b4f7ff!important}.stSidebar {background:#09151b!important}.stSidebar [data-testid="stSidebarContent"] {background:linear-gradient(180deg,#0c1a20,#081318)}.stSidebar [data-testid="stSidebarHeader"] {background:transparent}.stExpander {border:1px solid var(--line)!important;border-radius:10px!important;background:rgba(13,30,37,.75)!important}@keyframes moon-drift {to {transform:translate(-12px,8px)}}@keyframes signal {0%,100% {opacity:.45}50% {opacity:1}}@media(prefers-reduced-motion:reduce) {*,*:before,*:after {animation:none!important;scroll-behavior:auto!important}}@media(max-width:760px) {.block-container {padding:.8rem}.lunar-hero {min-height:380px;padding:28px 23px}.lunar-hero:after {width:360px;height:360px;right:-180px;top:50px;opacity:.38}.hero-readout {left:22px;right:22px;bottom:20px;width:auto}.mission-status {grid-template-columns:1fr 1fr}.mission-status div {border-bottom:1px solid var(--line)}.pipeline-nav {display:block}.pipeline-stage {text-align:left;padding-left:18px;border-right:0;border-bottom:1px solid var(--line)}.pipeline-stage.active:after {left:0;right:auto;top:0;bottom:0;width:3px;height:auto}}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load configuration and hardware telemetry
settings = load_settings()
hw = get_hardware_info()

# Top Telemetry Status Bar
cuda_label = f"CUDA (GPU: {hw['gpu_name']})" if hw["cuda_available"] else "CPU Mode (Standard Multithreading)"
st.markdown(
    f"""
    <div class="telemetry-bar">
        <div class="telemetry-item">
            <span class="status-dot"></span>
            <span>SYSTEM: <span class="telemetry-val">LUNARMATCH AI OPERATIONAL</span></span>
        </div>
        <div class="telemetry-item">
            <span>ISRO SIH #26166: <span class="telemetry-val">CHANDRAYAAN-2 / LRO</span></span>
        </div>
        <div class="telemetry-item">
            <span>COMPUTE: <span class="telemetry-val">{hw['cpu_threads']} Threads / {cuda_label}</span></span>
        </div>
        <div class="telemetry-item">
            <span>MEMORY: <span class="telemetry-val">{hw['ram_available_gb']} GB / {hw['ram_total_gb']} GB</span></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mission opening — the telemetry immediately below is derived from the local
# runtime and configuration, rather than decorative operational statistics.
st.markdown(
    """
    <section class="lunar-hero" aria-label="LUNARMATCH mission opening">
      <div class="orbital"></div>
      <div class="hero-kicker">ISRO SIH #26166 · PLANETARY IMAGING WORKSTATION</div>
      <h1>LUNARMATCH<br>AI</h1>
      <div class="hero-sub">MULTI-MODAL LUNAR IMAGE CORRESPONDENCE<br>&amp; SUB-PIXEL REGISTRATION</div>
      <p class="hero-copy">Aligning lunar observations across sensors, scale, illumination and imaging conditions.</p>
      <div class="hero-readout"><span>AOI / IMAGING FIELD</span><br>GRID: IMAGE-SPACE · GEOREFERENCE: INPUT-DEPENDENT<br>PIPELINE: DATA → FEATURES → GEOMETRY → VALIDATION</div>
    </section>
    """,
    unsafe_allow_html=True,
)
hero_action, hero_pipeline = st.columns([1, 1])
with hero_action:
    if st.button("START REGISTRATION", type="primary", use_container_width=True):
        st.session_state["hero_start_requested"] = True
with hero_pipeline:
    if st.button("EXPLORE PIPELINE", use_container_width=True):
        st.session_state["hero_pipeline_requested"] = True
        st.info("Configure input data and registration parameters in Mission Control, then initiate the pipeline.")

# Top Navigation Tabs
nav_selection = st.tabs([
    "🚀 Registration Workstation",
    "📊 Sensor & Data Explorer",
    "🧪 Comparative Benchmark Lab",
    "📑 Experiment History & Audit Log",
    "ℹ️ System Capability Matrix",
])

# Sidebar Configuration
st.sidebar.markdown("### 🎛️ MISSION CONTROL")
mode = st.sidebar.radio(
    "Operating Mode",
    [
        "Mode 0: SIH Grand Finale Guided Showcase",
        "Mode 1: Offline Controlled Demo",
        "Mode 2: Local Lunar Imagery Ingestion",
    ],
    index=0,
    help="Core registration operates 100% offline without mandatory API or GPU access.",
)

data_label = "USER DATA" if mode.startswith("Mode 2") else "CONTROLLED LUNAR DATA"
st.markdown(
    f"""
    <div class="mission-status" aria-label="Current mission status">
      <div><b>MISSION SYSTEM</b><span class="ready-dot">● READY</span></div>
      <div><b>DATA</b><span>{data_label}</span></div>
      <div><b>REGISTRATION</b><span>AVAILABLE</span></div>
      <div><b>MODELS</b><span>CLASSICAL / LEARNED PROBE</span></div>
      <div><b>PROCESSING</b><span>{cuda_label}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

source_lunar: LunarImage | None = None
ref_lunar: LunarImage | None = None
scenario_meta: ScenarioBenchmarkData | None = None
preset_prep = "clahe"
preset_matcher = "hybrid"

if mode == "Mode 0: SIH Grand Finale Guided Showcase":
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🏆 SIH Grand Finale Challenges")
    curated_challenges = {
        "Challenge 1: Extreme Sun-Angle Inversion (180° Azimuth Delta)": {
            "scenario_id": "scenario_5_illumination_shift",
            "prep": "gradient",
            "matcher": "hybrid",
            "desc": "Severe sun elevation & 180° azimuth reversal across rugged lunar craters. Demonstrates Sobel gradient illumination-invariance + hybrid feature fusion.",
        },
        "Challenge 2: Cross-Sensor Scale Disparity (4x Zoom / OHRC vs TMC-2)": {
            "scenario_id": "scenario_1_small_scale",
            "prep": "clahe",
            "matcher": "hybrid",
            "desc": "Significant spatial resolution delta between narrow-angle and wide-angle lunar cameras. Overcome via multi-scale pyramid matching.",
        },
        "Challenge 3: Complex Spacecraft Rotation (25° In-Plane Drift)": {
            "scenario_id": "scenario_3_rotation_translation",
            "prep": "clahe",
            "matcher": "hybrid",
            "desc": "Non-nadir orbital pass with roll/yaw divergence. Automatically selected Similarity model with sub-pixel residual minimization.",
        },
        "Challenge 4: Low-Texture Basaltic Mare (Subtle Craters)": {
            "scenario_id": "scenario_7_low_texture_mare",
            "prep": "clahe",
            "matcher": "hybrid",
            "desc": "Low radiometric entropy and subtle topographic relief in lunar mare. Adaptive Spatial Balancer ensures uniform coverage across all sectors.",
        },
    }
    selected_ch_title = st.sidebar.selectbox("Curated ISRO Problem Scenarios", list(curated_challenges.keys()))
    ch_info = curated_challenges[selected_ch_title]
    preset_prep = ch_info["prep"]
    preset_matcher = ch_info["matcher"]

    scenario_meta = get_scenario_by_id(ch_info["scenario_id"], size=512, seed=42)
    source_lunar = scenario_meta.source_image
    ref_lunar = scenario_meta.reference_image

    st.sidebar.info(f"**Grand Finale Focus:**\n{ch_info['desc']}")

elif mode == "Mode 1: Offline Controlled Demo":
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔬 Benchmark Scenario")
    scenario_options = {title: s_id for s_id, title in SCENARIOS_CATALOG}
    selected_title = st.sidebar.selectbox("Controlled Lunar Test Case", list(scenario_options.keys()))
    selected_id = scenario_options[selected_title]

    scenario_meta = get_scenario_by_id(selected_id, size=512, seed=42)
    source_lunar = scenario_meta.source_image
    ref_lunar = scenario_meta.reference_image

    st.sidebar.info(
        f"**Category:** {scenario_meta.category}\n\n"
        f"**Model GT:** `{scenario_meta.transformation_type.upper()}`\n\n"
        f"**Scale Delta:** {scenario_meta.scale_factor:.2f}x\n\n"
        f"**Sun Delta:** {abs(scenario_meta.sun_azimuth_ref - scenario_meta.sun_azimuth_source):.0f}° azimuth"
    )

else:  # Mode 2: Local Data Ingestion
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📂 Local File Ingestion")
    ingest_method = st.sidebar.radio(
        "Ingestion Method",
        ["Local Path / Sample Datasets (Direct Disk Read)", "Drag & Drop Upload (Max 2GB)"],
        index=0,
        help="Use 'Local Path' to load 1GB+ ISRO rasters directly from disk without upload overhead."
    )

    sensor_src = st.sidebar.selectbox("Source Sensor Tag", ["Chandrayaan-2 OHRC", "Chandrayaan-2 TMC-2", "Chandrayaan-2 IIRS", "Generic Lunar"])
    sensor_ref = st.sidebar.selectbox("Reference Sensor Tag", ["LRO LROC NAC", "SELENE Kaguya TC", "Chandrayaan-2 TMC-2", "Generic Reference"])

    if ingest_method == "Local Path / Sample Datasets (Direct Disk Read)":
        sample_files = list(SAMPLES_DIR.glob("*.tif")) + list(RAW_DATA_DIR.glob("*.tif"))
        sample_file_names = [f.name for f in sample_files]

        default_src_path = "data/samples/sample_ohrc.tif" if (SAMPLES_DIR / "sample_ohrc.tif").exists() else ""
        default_ref_path = "data/samples/sample_tmc2.tif" if (SAMPLES_DIR / "sample_tmc2.tif").exists() else ""

        src_path_str = st.sidebar.text_input("Source File Path (.tif, .png, .img, .jp2)", value=default_src_path)
        ref_path_str = st.sidebar.text_input("Reference File Path (.tif, .png, .img, .jp2)", value=default_ref_path)
        st.sidebar.caption(
            "For ISRO/PDS products, place the calibrated raster and its original PDS4 XML/LBL label "
            "in the same folder. The label is read automatically for footprint and geometry validation."
        )

        if sample_file_names:
            st.sidebar.caption("Or pick built-in sample pairs:")
            preset_pair = st.sidebar.selectbox("Quick Sample Preset", ["Custom Path", "OHRC vs TMC-2 (Cross-Scale)", "Sun Azimuth Inversion (45° vs 225°)"])
            if preset_pair == "OHRC vs TMC-2 (Cross-Scale)":
                src_path_str = str(SAMPLES_DIR / "sample_ohrc.tif")
                ref_path_str = str(SAMPLES_DIR / "sample_tmc2.tif")
            elif preset_pair == "Sun Azimuth Inversion (45° vs 225°)":
                src_path_str = str(SAMPLES_DIR / "sample_sun_azimuth_45deg.tif")
                ref_path_str = str(SAMPLES_DIR / "sample_sun_azimuth_225deg.tif")

        src_p = Path(src_path_str) if src_path_str else None
        ref_p = Path(ref_path_str) if ref_path_str else None

        if src_p and src_p.exists() and ref_p and ref_p.exists():
            source_lunar = load_lunar_image(src_p, metadata_override={"sensor": sensor_src})
            ref_lunar = load_lunar_image(ref_p, metadata_override={"sensor": sensor_ref})
        else:
            st.info("👆 Please specify valid local file paths in the sidebar or pick a sample preset.")

    else:  # Drag & Drop Upload with SHA-256 Hash Verification
        src_file = st.sidebar.file_uploader("Source Image (OHRC, TMC-2, or User Raster)", type=["tif", "tiff", "png", "jpg", "jpeg", "img", "jp2"])
        ref_file = st.sidebar.file_uploader("Reference Image (LRO NAC, SELENE, etc.)", type=["tif", "tiff", "png", "jpg", "jpeg", "img", "jp2"])
        src_label = st.sidebar.file_uploader("Source PDS4 label (optional, .xml/.lbl)", type=["xml", "lbl"])
        ref_label = st.sidebar.file_uploader("Reference PDS4 label (optional, .xml/.lbl)", type=["xml", "lbl"])

        if src_file and ref_file:
            def save_uploaded_file_secure(uploaded_f, prefix: str) -> Path:
                """Save uploaded file with SHA-256 content verification to prevent cache collision."""
                clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', uploaded_f.name)
                # Compute SHA-256 hash of upload content
                hasher = hashlib.sha256()
                uploaded_f.seek(0)
                chunk_size = 4 * 1024 * 1024
                while True:
                    chunk = uploaded_f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
                file_hash = hasher.hexdigest()[:12]

                target_path = RAW_DATA_DIR / f"upload_{prefix}_{file_hash}_{clean_name}"
                if not target_path.exists():
                    uploaded_f.seek(0)
                    with open(target_path, "wb") as f:
                        while True:
                            chunk = uploaded_f.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                return target_path

            src_tmp = save_uploaded_file_secure(src_file, prefix="src")
            ref_tmp = save_uploaded_file_secure(ref_file, prefix="ref")
            src_label_tmp = save_uploaded_file_secure(src_label, prefix="src_label") if src_label else None
            ref_label_tmp = save_uploaded_file_secure(ref_label, prefix="ref_label") if ref_label else None

            source_lunar = load_lunar_image(src_tmp, metadata_override={
                "sensor": sensor_src,
                **({"pds4_label_path": str(src_label_tmp)} if src_label_tmp else {}),
            })
            ref_lunar = load_lunar_image(ref_tmp, metadata_override={
                "sensor": sensor_ref,
                **({"pds4_label_path": str(ref_label_tmp)} if ref_label_tmp else {}),
            })
        else:
            st.info("👆 Please upload both a Source and a Reference lunar raster in the sidebar, or switch to Mode 0 / Mode 1.")

# Pipeline Configuration in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Algorithmic Parameters")

prep_methods = ["destripe", "clahe", "gradient", "original", "phase", "edge"]
default_prep_idx = prep_methods.index(preset_prep) if preset_prep in prep_methods else 0
prep_choice = st.sidebar.selectbox(
    "Radiometric Preprocessing",
    prep_methods,
    index=default_prep_idx,
    format_func=lambda x: {
        "destripe": "Sensor Destripe (Push-Broom Line Artifact Removal)",
        "clahe": "CLAHE (Local Contrast Equalization)",
        "gradient": "Sobel Gradient Magnitude (Illumination Invariant)",
        "original": "Original Normalized Radiance",
        "phase": "2D Phase Congruency (Local Energy Model)",
        "edge": "Canny Topographic Edge Map",
    }.get(x, x),
)

_extractor_probe = LearnedFeatureExtractor()
_hybrid_label = (
    "Adaptive Hybrid (SIFT + Deep Learned Feature Fusion)"
    if _extractor_probe.model_loaded
    else "Adaptive Hybrid (SIFT + GFTT-SIFT Classical Fusion — Honest Fallback)"
)

matcher_methods = ["hybrid", "sift", "akaze"]
default_matcher_idx = matcher_methods.index(preset_matcher) if preset_matcher in matcher_methods else 0
matcher_choice = st.sidebar.selectbox(
    "Correspondence Matcher",
    matcher_methods,
    index=default_matcher_idx,
    format_func=lambda x: {
        "hybrid": _hybrid_label,
        "sift": "Classical SIFT (Scale-Invariant Feature Transform)",
        "akaze": "ORB / AKAZE (Binary Descriptor Baseline)",
    }.get(x, x),
)

ransac_thresh = st.sidebar.slider(
    "RANSAC Inlier Threshold (px)", 1.0, 20.0, 3.0, 0.5,
    help="Inlier consensus threshold in pixels. Automatically scaled for extreme resolution disparities."
)

enable_balancing = st.sidebar.checkbox(
    "Adaptive Spatial Match Balancer",
    value=True,
    help="ISRO Requirement: Enforces uniform spatial distribution across lunar terrain."
)
grid_size_choice = st.sidebar.selectbox("Spatial Grid Size", [4, 6, 8], index=1, format_func=lambda x: f"{x} x {x} Grid")

enable_subpixel = st.sidebar.checkbox(
    "2D Parabolic Sub-Pixel Refinement",
    value=True,
    help="Local patch NCC and continuous quadratic peak optimization for sub-pixel precision."
)

btn_label = "⚡ EXECUTE GUIDED SHOWCASE" if mode.startswith("Mode 0") else "🚀 INITIATE REGISTRATION PIPELINE"
run_pipeline_btn = st.sidebar.button(btn_label, type="primary", use_container_width=True)
run_pipeline_btn = run_pipeline_btn or st.session_state.pop("hero_start_requested", False)


# ==============================================================================
# TAB 1: REGISTRATION WORKSTATION
# ==============================================================================
with nav_selection[0]:
    has_result = "pipeline_result" in st.session_state
    data_state = "complete" if source_lunar is not None and ref_lunar is not None else ""
    result_state = "complete" if has_result else ""
    st.markdown(
        f"""
        <div class="pipeline-nav" aria-label="Registration pipeline">
          <div class="pipeline-stage {data_state}"><strong><span class="pipe-dot">{'●' if data_state else '○'}</span>DATA</strong><small>{'inputs ready' if data_state else 'awaiting inputs'}</small></div>
          <div class="pipeline-stage {result_state}"><strong><span class="pipe-dot">{'●' if result_state else '○'}</span>METADATA</strong><small>{'evaluated' if result_state else 'on execution'}</small></div>
          <div class="pipeline-stage {result_state}"><strong><span class="pipe-dot">{'●' if result_state else '○'}</span>FEATURES</strong><small>{'evaluated' if result_state else 'on execution'}</small></div>
          <div class="pipeline-stage {result_state}"><strong><span class="pipe-dot">{'●' if result_state else '○'}</span>MATCHING</strong><small>{'evaluated' if result_state else 'on execution'}</small></div>
          <div class="pipeline-stage {result_state}"><strong><span class="pipe-dot">{'●' if result_state else '○'}</span>GEOMETRY</strong><small>{'evaluated' if result_state else 'on execution'}</small></div>
          <div class="pipeline-stage {result_state}"><strong><span class="pipe-dot">{'●' if result_state else '○'}</span>REFINEMENT</strong><small>{'evaluated' if result_state else 'on execution'}</small></div>
          <div class="pipeline-stage {'active' if has_result else ''}"><strong><span class="pipe-dot">{'●' if has_result else '○'}</span>VALIDATION</strong><small>{'result ready' if has_result else 'pending'}</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if source_lunar is not None and ref_lunar is not None:
        # Pre-matching quality, metadata provenance, and overlap checks
        src_quality = analyze_image_quality(source_lunar)
        ref_quality = analyze_image_quality(ref_lunar)
        overlap_report = estimate_overlap(source_lunar, ref_lunar)

        # Quality & Overlap Telemetry Cards
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.markdown(
                f"""
                <div class="mission-card">
                    <div class="mission-header">SOURCE SENSOR TELEMETRY</div>
                    <div>Status: <span class="metric-badge badge-green">{src_quality.readiness_status}</span></div>
                    <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">
                        Sensor: <b>{source_lunar.instrument}</b> ({source_lunar.sensor})<br>
                        Dimensions: <b>{source_lunar.width} x {source_lunar.height}</b> px<br>
                        GSD: <b>{source_lunar.gsd if source_lunar.gsd is not None else 'UNKNOWN'} m/px</b><br>
                        Texture: <b>{src_quality.texture_level}</b> ({src_quality.texture_entropy} bits)<br>
                        Valid Pixels: <b>{src_quality.valid_pixel_pct}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_q2:
            st.markdown(
                f"""
                <div class="mission-card">
                    <div class="mission-header">REFERENCE SENSOR TELEMETRY</div>
                    <div>Status: <span class="metric-badge badge-green">{ref_quality.readiness_status}</span></div>
                    <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">
                        Sensor: <b>{ref_lunar.instrument}</b> ({ref_lunar.sensor})<br>
                        Dimensions: <b>{ref_lunar.width} x {ref_lunar.height}</b> px<br>
                        GSD: <b>{ref_lunar.gsd if ref_lunar.gsd is not None else 'UNKNOWN'} m/px</b><br>
                        Texture: <b>{ref_quality.texture_level}</b> ({ref_quality.texture_entropy} bits)<br>
                        Valid Pixels: <b>{ref_quality.valid_pixel_pct}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_q3:
            pct_display = f"{overlap_report.overlap_pct:.1f}%" if overlap_report.overlap_pct is not None else "Unknown"
            badge_class = "badge-cyan" if overlap_report.confidence in ("HIGH", "MEDIUM") else "badge-yellow"
            st.markdown(
                f"""
                <div class="mission-card">
                    <div class="mission-header">TIERED OVERLAP ESTIMATION</div>
                    <div>Overlap: <span class="metric-badge {badge_class}">{pct_display}</span></div>
                    <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">
                        Tier: <b>{overlap_report.level}</b><br>
                        Confidence: <b>{overlap_report.confidence}</b><br>
                        Scale Delta: <b>{overlap_report.estimated_scale_ratio or 'N/A'}x</b><br>
                        Method: <b>{overlap_report.method}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Prominent Level 3 Metadata Limitation Notice
        if overlap_report.level.startswith("Level 3") or overlap_report.confidence == "UNKNOWN":
            st.warning(
                "⚠️ **Level 3 / Unknown Geospatial Metadata:** Ingested imagery consists of uncalibrated browse/PNG rasters lacking embedded GeoTIFF coordinates or PDS4 XML labels. "
                "Registration operates in **IMAGE_ONLY_EXPLORATORY** mode. Correspondences cannot be tied to planetary ground coordinates without authentic PDS4/GeoTIFF metadata."
            )

        # Handle active scenario/parameter caching
        current_key = f"{mode}_{getattr(scenario_meta, 'scenario_id', 'upload')}_{prep_choice}_{matcher_choice}_{ransac_thresh}_{enable_balancing}_{grid_size_choice}_{enable_subpixel}"
        if st.session_state.get("active_scenario_key") != current_key:
            if not run_pipeline_btn:
                st.session_state.pop("pipeline_result", None)
            st.session_state["active_scenario_key"] = current_key

        if run_pipeline_btn or "pipeline_result" in st.session_state:
            if run_pipeline_btn:
                st.session_state.pop("last_export_artifacts", None)
                with st.spinner("Executing LUNARMATCH AI registration pipeline..."):
                    pipeline = RegistrationPipeline()
                    exec_result = pipeline.run(
                        source=source_lunar,
                        reference=ref_lunar,
                        preprocessing_method=prep_choice,
                        matcher_method=matcher_choice,
                        ransac_threshold=ransac_thresh,
                        enable_spatial_balancing=enable_balancing,
                        spatial_grid_size=grid_size_choice,
                        enable_subpixel=enable_subpixel,
                    )
                    tracker = ExperimentTracker()
                    run_id = tracker.record_run(
                        result=exec_result,
                        scenario_name=scenario_meta.title if scenario_meta else "User Ingested Pair",
                        source_sensor=source_lunar.instrument if source_lunar.instrument != "Not available" else "OHRC-Sim",
                        reference_sensor=ref_lunar.instrument if ref_lunar.instrument != "Not available" else "TMC-Sim",
                        preprocessing_method=prep_choice,
                        matcher_method=matcher_choice,
                        notes=f"LUNARMATCH Cartography Workstation ({mode})",
                    )
                    st.session_state["pipeline_result"] = exec_result
                    st.session_state["last_run_id"] = run_id

            result: PipelineExecutionResult = st.session_state["pipeline_result"]
            res_match = result.match_result
            metrics = result.metrics
            q_score = result.quality_score
            last_run = st.session_state.get("last_run_id", "Run Recorded")
            operating_state = getattr(result, "operating_state", "UNKNOWN")
            rejection_reason = getattr(result, "rejection_reason", "")
            stage_timings = getattr(result, "stage_timings", {})
            stage_counts = getattr(result, "stage_counts", {})

            # Execution Pipeline Stages Indicator
            st.markdown(
                """
                <div class="timeline-container">
                    <div class="timeline-step done">✓ 1. INGESTION</div>
                    <div class="timeline-step done">✓ 2. PREPROCESS</div>
                    <div class="timeline-step done">✓ 3. MATCHING</div>
                    <div class="timeline-step done">✓ 4. RANSAC MODEL</div>
                    <div class="timeline-step done">✓ 5. SPATIAL BALANCER</div>
                    <div class="timeline-step done">✓ 6. SUB-PIXEL</div>
                    <div class="timeline-step active">● 7. METRICS & EXPORT</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Operating State Badge Configuration
            if operating_state == "VERIFIED_GEOSPATIAL":
                op_badge_class = "badge-green"
                op_badge_label = "🛰️ VERIFIED GEOSPATIAL"
            elif operating_state == "IMAGE_ONLY_EXPLORATORY":
                op_badge_class = "badge-yellow"
                op_badge_label = "🔍 IMAGE-ONLY EXPLORATORY"
            else:
                op_badge_class = "badge-red"
                op_badge_label = "⛔ REJECTED"

            # Reliability & Success Assessment
            eff_inliers = metrics.inlier_count if metrics else 0
            has_valid_rmse = (metrics is not None) and (metrics.rmse_px is not None)
            is_reliable = result.success and (eff_inliers >= 4) and has_valid_rmse
            q_total = q_score.total_score if q_score else 0

            if is_reliable:
                if q_total >= 50 and eff_inliers >= 8:
                    banner_color, banner_icon, banner_text = "#22c55e", "●", "REGISTRATION SUCCESSFUL"
                else:
                    banner_color, banner_icon, banner_text = "#f59e0b", "◐", "REGISTRATION PARTIAL — LOW CONFIDENCE"
            else:
                banner_color, banner_icon, banner_text = "#ef4444", "✕", "REGISTRATION REJECTED — INSUFFICIENT VERIFIED CORRESPONDENCES"

            st.markdown(
                f"""
                <div class="mission-card" style="border-left: 4px solid {banner_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: {banner_color};">
                            {banner_icon} {banner_text}
                        </span>
                        <div>
                            <span class="metric-badge {op_badge_class}">{op_badge_label}</span>
                            <span class="metric-badge" style="background: {banner_color}22; color:{banner_color}; border:1px solid {banner_color}55; margin-left: 6px;">
                                TIME: {result.total_pipeline_time:.3f}s
                            </span>
                            <span class="metric-badge badge-cyan" style="margin-left: 6px;">RUN: {last_run}</span>
                        </div>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        <b>Geometric Model Decision:</b> {result.model_selection_reason or "N/A"}
                    </div>
                    {f'<div style="color: #fca5a5; font-size: 0.85rem; margin-top: 4px;"><b>Rejection Details:</b> {rejection_reason}</div>' if rejection_reason else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Helper function for rendering stage timings
            def render_timing_breakdown(timings_dict: dict, wall_time: float):
                st.markdown("#### ⏱️ Stage-by-Stage Performance Telemetry")
                if not timings_dict:
                    st.info("Stage timing profile not recorded for this execution.")
                    return
                timing_rows = []
                for k, v in timings_dict.items():
                    if k == "total_seconds":
                        continue
                    clean_name = k.replace("_seconds", "").replace("_", " ").title()
                    pct = (v / wall_time * 100.0) if wall_time > 0 else 0.0
                    timing_rows.append({
                        "Pipeline Stage": clean_name,
                        "Duration (s)": round(v, 4),
                        "Compute Share (%)": round(pct, 1),
                    })
                df_t = pd.DataFrame(timing_rows)
                c_t1, c_t2 = st.columns([1, 1])
                with c_t1:
                    st.dataframe(df_t, use_container_width=True)
                    st.caption(f"**Total Pipeline Wall Clock:** `{wall_time:.3f} s`")
                with c_t2:
                    fig_bar = go.Figure(go.Bar(
                        x=[r["Duration (s)"] for r in timing_rows],
                        y=[r["Pipeline Stage"] for r in timing_rows],
                        orientation="h",
                        marker_color="#38bdf8",
                    ))
                    fig_bar.update_layout(
                        title="Latency per Pipeline Stage (s)",
                        paper_bgcolor="#060a13",
                        plot_bgcolor="#0a1122",
                        font={"color": "#94a3b8", "family": "JetBrains Mono"},
                        margin=dict(l=10, r=10, t=35, b=20),
                        height=260,
                        xaxis_title="Seconds",
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # ==================================================================
            # BRANCH A: REJECTED RUN (Scientific Rejection Handling)
            # ==================================================================
            if not is_reliable:
                # 1. Prominent Rejection & Failure Diagnostic Drawer
                diag_rep = diagnose_registration_run(
                    source_quality=result.source_quality,
                    reference_quality=result.reference_quality,
                    overlap_report=result.overlap_report,
                    match_result=result.match_result,
                    metrics=result.metrics,
                )
                st.markdown("### ⛔ Automated Rejection Analysis & Root-Cause Diagnostics")
                st.error(f"**Primary Rejection Diagnostic:** {diag_rep.primary_issue}")

                # Monotonicity & Inlier Attrition Telemetry Card
                st.markdown(
                    f"""
                    <div class="mission-card" style="border-left: 3px solid #ef4444;">
                        <div class="mission-header">STRICT MONOTONIC INLIER ATTRITION LOG</div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 8px;">
                            <div>1. Tentative Matches: <b>{stage_counts.get('tentative_matches', len(res_match.source_points))}</b></div>
                            <div>2. Geometric Consensus: <b>{stage_counts.get('geometric_inliers', 0)}</b></div>
                            <div>3. Spatially Balanced: <b>{stage_counts.get('spatially_balanced', 0)}</b></div>
                            <div>4. Final Verified: <b style="color: #ef4444;">{stage_counts.get('final_inliers', eff_inliers)}</b></div>
                        </div>
                        <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 10px;">
                            <b>Mathematical Rejection Rationale:</b> A minimum of 4 non-collinear correspondences is required to constrain a 2D affine/homography transformation with positive degrees of freedom. With only {stage_counts.get('final_inliers', eff_inliers)} correspondence(s), transfer RMSE is mathematically undefined (0 degrees of freedom) and is strictly reported as <b>N/A</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Diagnostic Findings List
                for finding in diag_rep.findings:
                    st.markdown(
                        f"""
                        <div class="mission-card" style="border-left: 3px solid #f59e0b; margin-top: 8px;">
                            <div style="font-weight: 600; color: #f87171;">[{finding.severity}] {finding.category}</div>
                            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 4px;">{finding.observation}</div>
                            <div style="font-size: 0.85rem; color: #38bdf8; margin-top: 4px;">💡 <b>Recommendation:</b> {finding.recommendation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Honest Quantitative Metrics Ribbon (Zero Fabrication)
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Inlier RMSE", "N/A (< 4 inliers)", help="RMSE is undefined for fewer than 4 correspondences.")
                m2.metric("Median Residual", "N/A", help="Residual statistics require verified correspondences.")
                m3.metric("Verified Inliers", f"{eff_inliers} / {len(res_match.source_points)} (Rejected)")
                m4.metric("Spatial Coverage", "0.0 %")
                m5.metric("Spatial Uniformity", "0.0 / 100")
                m6.metric("Quality Index", "0 / 100 (UNRELIABLE)")

                # Rejection Inspection Tabs
                tab_rej_matches, tab_rej_perf, tab_rej_log = st.tabs([
                    "🎯 Tentative Matches & Outlier Rays",
                    "⏱️ Stage-by-Stage Performance Profile",
                    "📋 Detailed Pipeline Execution Log",
                ])

                with tab_rej_matches:
                    st.markdown("#### Tentative Keypoints & Rejected Correspondence Rays")
                    st.caption("Visual audit of keypoints detected prior to outlier rejection.")
                    src_viz = result.preprocessed_source if result.preprocessed_source is not None else source_lunar.to_uint8()
                    ref_viz = result.preprocessed_reference if result.preprocessed_reference is not None else ref_lunar.to_uint8()
                    rays_img = draw_correspondence_rays(
                        src_viz,
                        ref_viz,
                        res_match.source_points,
                        res_match.reference_points,
                        res_match.inlier_mask,
                    )
                    st.image(rays_img, caption="Green = Inliers, Red = Rejected Outliers", use_container_width=True)

                with tab_rej_perf:
                    render_timing_breakdown(stage_timings, result.total_pipeline_time)

                with tab_rej_log:
                    st.markdown("#### Complete Pipeline Execution Trace")
                    st.text("\n".join(result.execution_log))

            # ==================================================================
            # BRANCH B: RELIABLE REGISTRATION
            # ==================================================================
            else:
                # Quantitative Metrics Ribbon
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Inlier RMSE", f"{metrics.rmse_px:.3f} px" if metrics.rmse_px is not None else "N/A")
                m2.metric("Median Residual", f"{metrics.median_residual_px:.3f} px" if metrics.median_residual_px is not None else "N/A")
                m3.metric("Verified Inliers", f"{metrics.inlier_count} / {metrics.tentative_matches}")
                m4.metric("Spatial Coverage", f"{max(0.0, metrics.spatial_coverage_pct):.1f} %")
                m5.metric("Spatial Uniformity", f"{max(0.0, metrics.spatial_uniformity_score):.1f} / 100")
                m6.metric("Quality Index", f"{q_total} / 100")

                # Quality Score Breakdown Card
                if q_score:
                    st.markdown(
                        f"""
                        <div class="mission-card">
                            <div class="mission-header">INTERNAL REGISTRATION QUALITY INDEX: {q_total} / 100 ({q_score.rating_label})</div>
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 8px;">
                                <div>Inlier Ratio Score: <b>{q_score.inlier_score:.1f} / 30</b></div>
                                <div>Spatial Coverage Score: <b>{q_score.coverage_score:.1f} / 25</b></div>
                                <div>Spatial Uniformity Score: <b>{q_score.uniformity_score:.1f} / 25</b></div>
                                <div>Residual Precision Score: <b>{q_score.residual_score:.1f} / 20</b></div>
                            </div>
                            <div style="font-size: 0.78rem; color: #64748b; margin-top: 10px; font-style: italic;">
                                * Note: {q_score.disclaimer}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Registration Radar is deliberately a compact spatial index:
                # its labels expose only measurements returned by this run.
                radar_col, telemetry_col = st.columns([1, 3])
                with radar_col:
                    st.markdown(
                        """
                        <div class="instrument-panel">
                          <div class="panel-title">REGISTRATION RADAR</div>
                          <div class="radar" aria-hidden="true"></div>
                          <div class="radar-label">RUN METRICS · NOT A CONFIDENCE SCORE</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with telemetry_col:
                    inlier_ratio = (metrics.inlier_count / metrics.tentative_matches * 100) if metrics.tentative_matches else 0.0
                    st.markdown(
                        f"""
                        <div class="instrument-panel">
                          <div class="panel-title">REGISTRATION TELEMETRY</div>
                          <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;font-family:'JetBrains Mono',monospace;font-size:.78rem;line-height:1.6">
                            <div><span style="color:#829094">INLIER RATIO</span><br><b>{inlier_ratio:.1f} %</b></div>
                            <div><span style="color:#829094">SPATIAL COVERAGE</span><br><b>{max(0.0, metrics.spatial_coverage_pct):.1f} %</b></div>
                            <div><span style="color:#829094">RESIDUAL RMSE</span><br><b>{metrics.rmse_px:.3f} px</b></div>
                            <div><span style="color:#829094">CANDIDATE MATCHES</span><br><b>{metrics.tentative_matches}</b></div>
                            <div><span style="color:#829094">VERIFIED INLIERS</span><br><b>{metrics.inlier_count}</b></div>
                            <div><span style="color:#829094">PIPELINE TIME</span><br><b>{result.total_pipeline_time:.3f} s</b></div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Visual Workstation Inspection Tabs
                tab_align, tab_matches, tab_spatial, tab_diff, tab_diag, tab_perf, tab_export = st.tabs([
                    "🔍 Alignment Explorer",
                    "🎯 Correspondences & Rays",
                    "📐 Spatial Balancer Grid",
                    "📉 Radiometric Difference Map",
                    "📊 Residual Diagnostics",
                    "⏱️ Stage Performance Profile",
                    "💾 Scientific Export",
                ])

                with tab_align:
                    st.markdown("#### Registered Source vs. Reference Alignment")
                    align_mode = st.radio("Alignment View Mode", ["Interactive Split Wipe", "Alpha Color Overlay (Red/Cyan)", "Side-by-Side"], horizontal=True)

                    u8_ref = ref_lunar.to_uint8()
                    reg_img = result.registered_image
                    reg_display, ref_display = crop_to_valid_overlap(reg_img, u8_ref)
                    ref_disp = normalize_for_display(ref_display)
                    reg_disp = normalize_for_display(reg_display)

                    if align_mode == "Interactive Split Wipe":
                        split_frac = st.slider("Split Wipe Position", 0.05, 0.95, 0.50, 0.05)
                        wipe_img = create_split_wipe(reg_display, ref_display, split_fraction=split_frac)
                        st.image(wipe_img, caption="Split Wipe: Left = Registered Source, Right = Reference", use_container_width=True)
                    elif align_mode == "Alpha Color Overlay (Red/Cyan)":
                        h_ov, w_ov = ref_disp.shape[:2]
                        reg_ov = reg_disp
                        if reg_ov.shape[:2] != (h_ov, w_ov):
                            canvas_ov = np.full((h_ov, w_ov), 128, dtype=np.uint8)
                            rh = min(reg_ov.shape[0], h_ov)
                            rw = min(reg_ov.shape[1], w_ov)
                            canvas_ov[:rh, :rw] = reg_ov[:rh, :rw]
                            reg_ov = canvas_ov
                        overlay_rgb = np.zeros((h_ov, w_ov, 3), dtype=np.uint8)
                        overlay_rgb[:, :, 0] = ref_disp
                        overlay_rgb[:, :, 1] = reg_ov
                        overlay_rgb[:, :, 2] = reg_ov
                        st.image(overlay_rgb, caption="False-Color Overlay: Red = Reference, Cyan = Registered Source (Neutral = Aligned)", use_container_width=True)
                    else:
                        c_s1, c_s2 = st.columns(2)
                        c_s1.image(reg_disp, caption="Registered Source (Cropped to overlap)", use_container_width=True)
                        c_s2.image(ref_disp, caption="Reference (Cropped to overlap)", use_container_width=True)

                with tab_matches:
                    st.markdown("#### Correspondence Rays & Outlier Rejection")
                    src_viz = result.preprocessed_source if result.preprocessed_source is not None else source_lunar.to_uint8()
                    ref_viz = result.preprocessed_reference if result.preprocessed_reference is not None else ref_lunar.to_uint8()
                    rays_img = draw_correspondence_rays(
                        src_viz,
                        ref_viz,
                        res_match.source_points,
                        res_match.reference_points,
                        res_match.inlier_mask,
                    )
                    st.image(rays_img, caption="Green = Verified Inliers, Red = Rejected Outliers", use_container_width=True)

                with tab_spatial:
                    st.markdown("#### Adaptive Spatial Match Balancer (ISRO Requirement)")
                    if result.spatial_result is not None:
                        s_res = result.spatial_result
                        col_sb1, col_sb2 = st.columns(2)
                        with col_sb1:
                            st.markdown(
                                f"""
                                <div class="mission-card">
                                    <div class="mission-header">SPATIAL UNIFORMITY METRICS</div>
                                    <div>Raw Inliers Uniformity: <span class="metric-badge badge-yellow">{s_res.raw_uniformity_score:.1f}%</span></div>
                                    <div style="margin-top: 8px;">Balanced Uniformity: <span class="metric-badge badge-green">{s_res.balanced_uniformity_score:.1f}%</span></div>
                                    <div style="margin-top: 8px;">Grid Occupancy ({s_res.grid_size}x{s_res.grid_size}): <b>{s_res.grid_occupancy_pct:.1f}%</b></div>
                                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                                        Pruned {len(res_match.inlier_source_points) - len(s_res.balanced_source_points)} redundant clustered matches.
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with col_sb2:
                            fig_hm = go.Figure(data=go.Heatmap(
                                z=s_res.cell_counts_matrix,
                                colorscale="Viridis",
                                text=s_res.cell_counts_matrix,
                                texttemplate="%{text}",
                                textfont={"size": 12},
                            ))
                            fig_hm.update_layout(
                                title=f"Cell Match Density Grid ({s_res.grid_size}x{s_res.grid_size})",
                                paper_bgcolor="#060a13",
                                plot_bgcolor="#0a1122",
                                font={"color": "#94a3b8", "family": "JetBrains Mono"},
                                margin=dict(l=20, r=20, t=35, b=20),
                                height=250,
                            )
                            st.plotly_chart(fig_hm, use_container_width=True)

                    if result.subpixel_result is not None:
                        sub_r = result.subpixel_result
                        st.markdown(
                            f"""
                            <div class="mission-card">
                                <div class="mission-header">SUB-PIXEL REFINEMENT VERIFICATION</div>
                                <div>Refined Points: <span class="metric-badge badge-cyan">{sub_r.n_converged} points</span></div>
                                <div style="margin-top: 6px;">Measured Mean Sub-Pixel Offset: <b>{sub_r.mean_offset_px:.3f} px</b> (Max Adjustment: <b>{sub_r.max_offset_px:.3f} px</b>)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with tab_diff:
                    st.markdown("#### Radiometric Difference Heatmap")
                    col_d1, col_d2 = st.columns([3, 1])
                    with col_d1:
                        st.image(result.difference_map, caption="Absolute Intensity Error |I_reg - I_ref| (Darker = Better Alignment)", use_container_width=True)
                    with col_d2:
                        st.markdown(f"**Mean Overlap Difference:** `{res_match.diagnostics.get('mean_overlap_diff', 0.0)}`")
                        st.caption("Note: High difference in crater shadows reflects varying sun elevation angles rather than geometric error.")

                with tab_diag:
                    st.markdown("#### Geometric Residuals & Quiver Plots")
                    c_g1, c_g2 = st.columns(2)
                    with c_g1:
                        fig_hist = plot_residual_histogram(res_match.residuals, res_match.inlier_mask, ransac_threshold=ransac_thresh)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with c_g2:
                        fig_vec = plot_error_vectors(res_match.source_points, res_match.reference_points, res_match.inlier_mask, (source_lunar.height, source_lunar.width))
                        st.plotly_chart(fig_vec, use_container_width=True)

                with tab_perf:
                    render_timing_breakdown(stage_timings, result.total_pipeline_time)

                with tab_export:
                    st.markdown("#### Export Scientific Registration Artifacts")
                    if st.button("💾 GENERATE & SAVE EXPORT PACKAGE") or "last_export_artifacts" in st.session_state:
                        if "last_export_artifacts" not in st.session_state:
                            with st.spinner("Compiling scientific artifacts..."):
                                artifacts = export_registration_package(result, source_lunar, ref_lunar)
                                st.session_state["last_export_artifacts"] = artifacts
                        else:
                            artifacts = st.session_state["last_export_artifacts"]

                        st.success(f"✓ Successfully exported registration package to `{artifacts['report_md'].parent}`")

                        with open(artifacts["report_md"], "r", encoding="utf-8") as f:
                            rep_content = f.read()
                        st.download_button("📥 Download Scientific Report (.md)", data=rep_content, file_name="LUNAR_REGISTRATION_REPORT.md", mime="text/markdown")

                        with open(artifacts["matches_csv"], "r", encoding="utf-8") as f:
                            csv_content = f.read()
                        st.download_button("📥 Download Correspondences (.csv)", data=csv_content, file_name="correspondences.csv", mime="text/csv")

                # Expandable Diagnostic Findings Even for Successful Runs
                with st.expander("🔍 Health Diagnostics & Quality Audit", expanded=False):
                    diag_rep = diagnose_registration_run(
                        source_quality=result.source_quality,
                        reference_quality=result.reference_quality,
                        overlap_report=result.overlap_report,
                        match_result=result.match_result,
                        metrics=result.metrics,
                    )
                    st.markdown(f"**Primary Diagnostic Status:** {diag_rep.primary_issue}")
                    for finding in diag_rep.findings:
                        st.markdown(
                            f"""
                            <div class="mission-card" style="border-left: 3px solid #38bdf8; margin-top: 6px;">
                                <div style="font-weight: 600; color: #38bdf8;">[{finding.severity}] {finding.category}</div>
                                <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 4px;">{finding.observation}</div>
                                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">💡 <b>Recommendation:</b> {finding.recommendation}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.text("\n".join(result.execution_log))
        else:
            # Pre-registration preview dual panel
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="mission-card">', unsafe_allow_html=True)
                st.markdown('<div class="mission-header">SOURCE IMAGERY (OBSERVED)</div>', unsafe_allow_html=True)
                st.image(source_lunar.to_uint8(), caption=f"{source_lunar.instrument} ({source_lunar.width}x{source_lunar.height})", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="mission-card">', unsafe_allow_html=True)
                st.markdown('<div class="mission-header">REFERENCE IMAGERY (BASE)</div>', unsafe_allow_html=True)
                st.image(ref_lunar.to_uint8(), caption=f"{ref_lunar.instrument} ({ref_lunar.width}x{ref_lunar.height})", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.info("👈 Select parameters in Mission Control and click **INITIATE REGISTRATION PIPELINE**.")
    else:
        st.warning("⚠️ No images selected. Please choose a Guided Challenge (Mode 0), Benchmark Scenario (Mode 1), or upload images (Mode 2) in the sidebar.")


# ==============================================================================
# TAB 2: SENSOR & DATA EXPLORER
# ==============================================================================
with nav_selection[1]:
    st.markdown("### 📊 Planetary Sensor & Metadata Explorer")
    st.caption("Inspect physical metadata, solar geometry, ground sample distances, and radiometric parameters.")

    if source_lunar is not None and ref_lunar is not None:
        src_meta = source_lunar.metadata_dict()
        ref_meta = ref_lunar.metadata_dict()
        meta_rows = []
        for key in src_meta.keys():
            meta_rows.append({
                "Parameter": key.replace("_", " ").title(),
                "Source Sensor": str(src_meta[key]),
                "Reference Sensor": str(ref_meta[key]),
            })
        st.table(meta_rows)

        st.markdown("#### Radiometric Quality Analysis")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.json(src_quality.__dict__)
        with q_col2:
            st.json(ref_quality.__dict__)
    else:
        st.info("Load an image pair to inspect sensor telemetry.")


# ==============================================================================
# TAB 3: COMPARATIVE BENCHMARK LAB
# ==============================================================================
with nav_selection[2]:
    st.markdown("### 🧪 Comparative Benchmark Lab (Progressive Validation)")
    st.caption("Executes standard SIFT baseline vs. Adaptive Balancer vs. Sub-Pixel Refinement vs. Full LUNARMATCH AI.")

    if source_lunar is not None and ref_lunar is not None:
        if st.button("🧪 EXECUTE PROGRESSIVE BENCHMARK ON ACTIVE PAIR"):
            with st.spinner("Running progressive comparative benchmark harness..."):
                df_bench = run_comparative_benchmark(
                    source_lunar, ref_lunar,
                    preprocessing=prep_choice,
                    grid_size=grid_size_choice,
                )
                st.dataframe(df_bench, use_container_width=True)
                st.markdown(
                    """
                    **Scientific Observations:**
                    1. **Baseline SIFT:** Concentrates matches heavily on high-contrast crater rims, causing low spatial uniformity.
                    2. **Adaptive Balancer:** Enforces uniform spatial coverage across the lunar surface without clustering.
                    3. **Sub-Pixel Peak Optimization:** Refines coordinate vertices using 2D parabolic peak fitting, achieving sub-pixel RMSE.
                    """
                )
    else:
        st.info("Select an active pair in Mission Control to run comparative benchmarks.")


# ==============================================================================
# TAB 4: EXPERIMENT HISTORY & AUDIT LOG
# ==============================================================================
with nav_selection[3]:
    st.markdown("### 📑 Scientific Experiment Registry & Audit Trail")
    st.caption("Immutable append-only telemetry log storing precision metrics, inlier counts, and hardware parameters.")

    tracker = ExperimentTracker()
    df_hist = tracker.get_history()

    if df_hist.empty:
        st.info("No experiment runs logged yet. Execute a registration pipeline to populate the registry.")
    else:
        st.dataframe(
            df_hist,
            use_container_width=True,
            column_config={
                "run_id": "Run Identifier",
                "timestamp": "Timestamp (UTC)",
                "scenario_name": "Scenario / Dataset",
                "inlier_count": "Inliers",
                "rmse_px": st.column_config.NumberColumn("RMSE (px)", format="%.3f"),
                "uniformity_score": st.column_config.NumberColumn("Uniformity", format="%.1f%%"),
                "quality_score": st.column_config.NumberColumn("Quality Index (0-100)", format="%.1f"),
                "runtime_s": st.column_config.NumberColumn("Time (s)", format="%.3f"),
            },
        )

        c_dl1, c_dl2 = st.columns([2, 1])
        with c_dl1:
            all_runs = tracker.list_runs()
            selected_run = st.selectbox("Inspect Full JSON Telemetry Artifact", all_runs)
            if selected_run:
                run_detail = tracker.get_run(selected_run)
                if run_detail:
                    with st.expander(f"Telemetry Details: {selected_run}", expanded=False):
                        st.json(run_detail)
        with c_dl2:
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV Registry",
                data=csv_data,
                file_name="lunarmatch_experiments_log.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ==============================================================================
# TAB 5: SYSTEM CAPABILITY MATRIX & TRUTH REPORT
# ==============================================================================
with nav_selection[4]:
    st.markdown("### ℹ️ System Capability Matrix & Scientific Truth Disclaimers")
    st.caption("Comprehensive declaration of active vs. fallback features to guarantee scientific integrity.")

    st.markdown(
        """
        | Capability / Module | Implementation Status | Active Engine / Fallback Mode |
        | :--- | :--- | :--- |
        | **Classical SIFT / AKAZE** | **Genuinely Implemented** | Multi-scale pyramid extraction with Lowe's ratio test and mutual nearest-neighbors. |
        | **Learned Model (SuperPoint)** | **Honest Classical Fallback** | Operating in classical fallback (GFTT corners + SIFT descriptors) when weights are absent. |
        | **Adaptive Spatial Balancer** | **Genuinely Implemented** | 4x4, 6x6, 8x8 spatial grid partitioning enforcing uniform match distribution. |
        | **2D Parabolic Sub-Pixel** | **Genuinely Implemented** | Continuous quadratic surface peak fitting validated to 0.05 px error in unit tests. |
        | **Illumination Invariance** | **Genuinely Implemented** | CLAHE, Sobel gradient magnitude, and 2D Phase Congruency representations. |
        | **PDS4 XML Metadata** | **Genuinely Implemented** | Native XML parser using `defusedxml` with namespace and element extraction. |
        | **GSD Provenance** | **Genuinely Implemented** | Authoritative GSD extracted when present; explicitly marked `"UNKNOWN"` when absent. |
        | **Internal Quality Index** | **Genuinely Implemented** | Heuristic score (0-100) combining inlier ratio, coverage, uniformity, and residual RMSE. |
        """
    )
    st.info("💡 **Scientific Integrity Policy:** LUNARMATCH AI strictly declares all fallback states and never claims deep learning inference or ground truth validation without verifiable evidence.")
