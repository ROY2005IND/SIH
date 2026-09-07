"""Test suite for Phase 5 & 14: Scientific Evaluation Metrics & Quality Score."""
import numpy as np
import pytest
from evaluation.metrics import compute_registration_metrics, RegistrationMetrics
from evaluation.quality_score import calculate_quality_score
from evaluation.benchmark import run_comparative_benchmark
from demo.scenarios import get_scenario_by_id

def test_metrics_and_quality_score():
    """Verify quantitative metrics computation and explainable quality scoring."""
    shape = (200, 200)
    rng = np.random.default_rng(42)

    # 50 inliers with low residual (0.4 px) and 10 outliers
    inliers = rng.uniform(20.0, 180.0, (50, 2)).astype(np.float32)
    outliers = rng.uniform(0.0, 200.0, (10, 2)).astype(np.float32)
    ref_pts = np.vstack([inliers, outliers])
    src_pts = ref_pts.copy()

    inlier_mask = np.array([True] * 50 + [False] * 10, dtype=bool)
    residuals = np.array([0.4] * 50 + [25.0] * 10, dtype=np.float32)

    metrics = compute_registration_metrics(src_pts, ref_pts, inlier_mask, residuals, shape, runtime_seconds=0.12)
    assert metrics.inlier_count == 50
    assert pytest.approx(metrics.inlier_ratio_pct, 0.1) == 83.33
    assert pytest.approx(metrics.rmse_px, 0.01) == 0.40
    assert metrics.spatial_coverage_pct > 40.0
    assert metrics.spatial_uniformity_score > 40.0

    score = calculate_quality_score(metrics)
    assert score.total_score >= 70
    assert score.rating_label in ["EXEMPLARY", "RELIABLE"]
    assert "ISRO" in score.disclaimer

def test_comparative_benchmark_harness():
    """Verify benchmark engine executes and returns comparative DataFrame."""
    scenario = get_scenario_by_id("scenario_1_small_scale", size=128, seed=42)
    df = run_comparative_benchmark(scenario.source_image, scenario.reference_image, grid_size=4)

    assert len(df) == 4
    assert "Method" in df.columns
    assert "RMSE (px)" in df.columns
    assert "Quality Score" in df.columns
    assert "Uniformity (0-100)" in df.columns
