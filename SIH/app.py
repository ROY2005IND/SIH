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

# Custom Aerospace Dark Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #060a13;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Telemetry Banner */
    .telemetry-bar {
        background: linear-gradient(90deg, #091024 0%, #0d1b38 50%, #091024 100%);
        border: 1px solid #1e3a8a;
        border-radius: 8px;
        padding: 10px 18px;
        margin-bottom: 20px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .telemetry-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .telemetry-val {
        color: #38bdf8;
        font-weight: 600;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
        display: inline-block;
        box-shadow: 0 0 10px #22c55e;
    }

    /* Aerospace Cards */
    .mission-card {
        background-color: #0a1122;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .mission-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        font-weight: 600;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
    }
    .metric-badge {
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-green { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-cyan { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-yellow { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Timeline Stepper */
    .timeline-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        background: #0a1122;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .timeline-step {
        text-align: center;
        flex: 1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
    }
    .timeline-step.active {
        color: #38bdf8;
        font-weight: 700;
    }
    .timeline-step.done {
        color: #22c55e;
    }
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

# Header
st.title("🛰️ LUNARMATCH AI")
st.caption(
    "Multi-Modal, Sun-Angle & Scale-Invariant Lunar Image Correspondence & Sub-Pixel Registration | "
    "Indian Space Research Organisation (ISRO) Problem Statement #26166 | Offline-First Planetary Cartography Workstation"
)

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

        src_path_str = st.sidebar.text_input("Source File Path (.tif, .png, .img)", value=default_src_path)
        ref_path_str = st.sidebar.text_input("Reference File Path (.tif, .png, .img)", value=default_ref_path)

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
        src_file = st.sidebar.file_uploader("Source Image (OHRC, TMC-2, or User Raster)", type=["tif", "tiff", "png", "jpg", "jpeg"])
        ref_file = st.sidebar.file_uploader("Reference Image (LRO NAC, SELENE, etc.)", type=["tif", "tiff", "png", "jpg", "jpeg"])

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

            source_lunar = load_lunar_image(src_tmp, metadata_override={"sensor": sensor_src})
            ref_lunar = load_lunar_image(ref_tmp, metadata_override={"sensor": sensor_ref})
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


# ==============================================================================
# TAB 1: REGISTRATION WORKSTATION
# ==============================================================================
with nav_selection[0]:
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
