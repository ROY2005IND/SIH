"""
End-to-End System Integration & Regression Test Suite for MoonFlower AI.
Validates the complete scientific registration chain:
Data -> Radiometric Quality -> Overlap -> Preprocessing -> Hybrid Matching
-> Model Selection -> Spatial Balancing -> Subpixel Refinement -> Warping
-> Metrics -> Quality Score -> Diagnostics -> Export -> Experiment Tracking.
"""
import numpy as np
import pytest
from pathlib import Path

from demo.scenarios import get_scenario_by_id
from pipeline.registration_pipeline import RegistrationPipeline
from pipeline.export import export_registration_package
from experiments.tracker import ExperimentTracker
from analysis.failure_diagnostics import diagnose_registration_run
from preprocessing.hyperspectral import HyperspectralProcessor
from preprocessing.loader import LunarImage
from registration.non_rigid import analyze_residual_spatial_strain, warp_piecewise_affine
from evaluation.benchmark import run_comparative_benchmark

def test_full_lifecycle_pipeline(tmp_path: Path):
    """Execute complete end-to-end pipeline across all scientific modules."""
    scenario = get_scenario_by_id("scenario_1_small_scale", size=256, seed=42)
    pipeline = RegistrationPipeline()

    # 1. Pipeline Execution
    res = pipeline.run(
        source=scenario.source_image,
        reference=scenario.reference_image,
        preprocessing_method="clahe",
        matcher_method="sift",
        enable_spatial_balancing=True,
        spatial_grid_size=6,
        enable_subpixel=True,
    )

    assert res.success is True
    assert res.match_result is not None
    assert res.match_result.n_inliers >= 10
    assert res.metrics is not None
    assert res.metrics.rmse_px is not None and res.metrics.rmse_px < 3.0
    assert res.metrics.spatial_uniformity_score > 50.0
    assert res.spatial_result is not None
    assert res.subpixel_result is not None
    assert res.quality_score is not None
    assert res.quality_score.total_score >= 60

    # 2. Diagnostics
    diag = diagnose_registration_run(
        res.source_quality,
        res.reference_quality,
        res.overlap_report,
        res.match_result,
        sun_azimuth_delta_deg=20.0,
    )
    assert diag.is_success is True
    assert isinstance(diag.primary_issue, str)

    # 3. Export Package
    export_dir = tmp_path / "exports"
    artifacts = export_registration_package(
        res, scenario.source_image, scenario.reference_image, export_dir=export_dir
    )
    for key in ["matches_csv", "metrics_json", "transformation_json", "report_md"]:
        assert key in artifacts
        assert artifacts[key].exists()

    # 4. Experiment Tracking
    tracker = ExperimentTracker(runs_dir=tmp_path / "runs", results_dir=tmp_path / "results")
    run_id = tracker.record_run(
        result=res,
        scenario_name="Scenario 1 Lifecycle Test",
        source_sensor="OHRC",
        reference_sensor="TMC-2",
        notes="Full lifecycle automated integration test",
    )
    assert run_id.startswith("run_")
    history = tracker.get_history()
    assert len(history) == 1
    assert history.iloc[0]["scenario_name"] == "Scenario 1 Lifecycle Test"
    assert history.iloc[0]["selected_model"] == res.selected_model

def test_hyperspectral_cube_reduction_and_registration():
    """Verify IIRS 250-band hyperspectral cube reduction and subsequent registration."""
    h, w, bands = 128, 128, 50
    # Procedural lunar spectra (H, W, B)
    rng = np.random.default_rng(101)
    base_terrain = rng.uniform(0.1, 0.9, size=(h, w))
    spectral_curves = np.sin(np.linspace(0.1, np.pi - 0.1, bands))
    cube_data = (base_terrain[:, :, np.newaxis] * spectral_curves[np.newaxis, np.newaxis, :]).astype(np.float32)

    wavelengths = [round(float(w), 3) for w in np.linspace(0.8, 5.0, bands)]
    proc = HyperspectralProcessor(cube=cube_data, wavelengths=wavelengths)

    # 1. Band slicing by wavelength (e.g. 1.25 um)
    sliced_band, idx, actual_um = proc.extract_band_by_wavelength(1.25)
    assert sliced_band.shape == (h, w)
    assert sliced_band.dtype == np.uint8

    # 2. PCA reduction to PC1 (dominant albedo image)
    reduced_img = proc.get_principal_component(pc_index=0)
    assert reduced_img.shape == (h, w)
    assert reduced_img.dtype == np.uint8

    # Wrap in LunarImage
    source_iirs = LunarImage(
        image=reduced_img,
        width=w,
        height=h,
        channels=1,
        mission="Chandrayaan-2",
        instrument="IIRS",
        sensor="IIRS-PC1",
        wavelength="1.25 um / PC1",
    )
    ref_scenario = get_scenario_by_id("scenario_1_small_scale", size=128, seed=42)

    pipeline = RegistrationPipeline()
    res = pipeline.run(
        source=source_iirs,
        reference=ref_scenario.reference_image,
        preprocessing_method="clahe",
        matcher_method="sift",
    )
    # Registration executes safely even with synthetic noise
    assert res is not None
    assert hasattr(res, "success")

def test_residual_strain_and_piecewise_affine():
    """Verify Moran's I residual spatial strain detection and Delaunay piecewise affine warping."""
    source_pts = np.array([
        [20, 20], [80, 25], [140, 20],
        [25, 80], [85, 85], [145, 80],
        [20, 140], [80, 145], [140, 140],
    ], dtype=np.float32)

    # Add localized non-linear strain
    ref_pts = source_pts.copy()
    ref_pts[4] += np.array([12.0, 10.0], dtype=np.float32) # High local distortion at center

    residuals = np.linalg.norm(ref_pts - source_pts, axis=1)
    strain_rep = analyze_residual_spatial_strain(ref_pts, residuals)
    assert strain_rep is not None
    assert isinstance(strain_rep.spatial_autocorrelation, float)

    # Test piecewise affine warping
    src_img = (np.eye(160, dtype=np.uint8) * 200) + 50
    warped = warp_piecewise_affine(src_img, source_pts, ref_pts, (160, 160))
    assert warped.shape == (160, 160)
    assert np.any(warped > 0)

def test_comparative_benchmark_execution():
    """Verify comparative benchmark executes the 4 progressive pipelines and outputs clean table."""
    scenario = get_scenario_by_id("scenario_1_small_scale", size=128, seed=42)
    df = run_comparative_benchmark(
        source=scenario.source_image,
        reference=scenario.reference_image,
        preprocessing="clahe",
        grid_size=4,
    )

    assert len(df) == 4
    expected_methods = [
        "1. SIFT Baseline",
        "2. SIFT + Spatial Balancer",
        "3. SIFT + Sub-Pixel Refinement",
        "4. MOONFLOWER Full Pipeline",
    ]
    for m in expected_methods:
        assert m in df["Method"].values
    assert "Quality Score" in df.columns
    assert "RMSE (px)" in df.columns
    assert "Uniformity (0-100)" in df.columns
