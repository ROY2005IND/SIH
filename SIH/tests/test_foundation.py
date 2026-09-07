"""Test suite for Phase 0: Project foundation, environment, configuration, and hardware telemetry."""
import numpy as np
import pytest
from config import load_settings
from utils.paths import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, OUTPUTS_DIR
from utils.hardware import get_hardware_info
from utils.logging import get_logger
from matching.result import MatchResult

def test_config_loader():
    """Verify settings.yaml loads and contains necessary algorithm configurations."""
    settings = load_settings()
    assert isinstance(settings, dict)
    assert "preprocessing" in settings
    assert "matching" in settings
    assert "registration" in settings
    assert "spatial" in settings
    assert settings["spatial"]["default_grid"] in [4, 6, 8]

def test_paths_exist():
    """Verify directory structure exists and is properly isolated."""
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert OUTPUTS_DIR.exists()

def test_hardware_telemetry():
    """Verify hardware detection runs cleanly on CPU/GPU."""
    telemetry = get_hardware_info()
    assert "device" in telemetry
    assert telemetry["device"] in ["cpu", "cuda"]
    assert "cpu_cores" in telemetry
    assert telemetry["cpu_cores"] >= 1
    assert "ram_total_gb" in telemetry
    assert telemetry["ram_total_gb"] > 0

def test_aerospace_logger():
    """Verify structured logger formats and outputs cleanly."""
    logger = get_logger("TestLogger")
    logger.info("Telemetry system initialized.")
    assert logger.name == "MoonFlower.TestLogger"

def test_match_result_contract():
    """Verify MatchResult contract, metrics calculation, and properties."""
    src_pts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    ref_pts = np.array([[12.0, 22.0], [32.0, 42.0], [100.0, 120.0]], dtype=np.float32)
    confidences = np.array([0.9, 0.85, 0.4], dtype=np.float32)
    inlier_mask = np.array([True, True, False], dtype=bool)
    residuals = np.array([2.828, 2.828, 70.0], dtype=np.float32)

    res = MatchResult(
        method="sift",
        source_points=src_pts,
        reference_points=ref_pts,
        confidences=confidences,
        inlier_mask=inlier_mask,
        transformation=np.eye(3),
        transformation_type="affine",
        residuals=residuals,
        runtime_seconds=0.045,
    )

    assert res.n_matches == 3
    assert res.n_inliers == 2
    assert pytest.approx(res.inlier_ratio, 0.01) == 0.67
    assert res.rmse is not None
    assert pytest.approx(res.rmse, 0.01) == 2.83
    assert res.median_residual is not None
    assert pytest.approx(res.median_residual, 0.01) == 2.83
