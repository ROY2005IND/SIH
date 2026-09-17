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


def test_scientific_rejection_insufficient_inliers():
    """Verify that RMSE is strictly None and Quality is UNRELIABLE when inliers < 4."""
    shape = (256, 256)

    # Case A: Exactly 1 inlier (must NOT report 0.000 px RMSE)
    src_1 = np.array([[50.0, 50.0], [100.0, 100.0]], dtype=np.float32)
    ref_1 = np.array([[50.0, 50.0], [200.0, 200.0]], dtype=np.float32)
    mask_1 = np.array([True, False], dtype=bool)
    res_1 = np.array([0.0, 50.0], dtype=np.float32)

    metrics_1 = compute_registration_metrics(src_1, ref_1, mask_1, res_1, shape, runtime_seconds=0.05)
    assert metrics_1.inlier_count == 1
    assert metrics_1.rmse_px is None, "RMSE must be None for 1 inlier (0 degrees of freedom)"
    assert metrics_1.median_residual_px is None
    assert metrics_1.p90_residual_px is None
    assert metrics_1.max_inlier_error_px is None

    score_1 = calculate_quality_score(metrics_1)
    assert score_1.total_score == 0
    assert score_1.rating_label == "UNRELIABLE"

    # Case B: Exactly 3 inliers (insufficient to constrain 2D homography / affine)
    src_3 = np.array([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32)
    ref_3 = src_3.copy()
    mask_3 = np.array([True, True, True], dtype=bool)
    res_3 = np.array([0.1, 0.2, 0.15], dtype=np.float32)

    metrics_3 = compute_registration_metrics(src_3, ref_3, mask_3, res_3, shape, runtime_seconds=0.05)
    assert metrics_3.inlier_count == 3
    assert metrics_3.rmse_px is None, "RMSE must be None for 3 inliers"

    score_3 = calculate_quality_score(metrics_3)
    assert score_3.total_score == 0
    assert score_3.rating_label == "UNRELIABLE"


def test_collinear_degenerate_points():
    """Verify spatial coverage handles collinear points safely."""
    shape = (256, 256)
    # Collinear points on y = 50
    src_collinear = np.array([[10.0, 50.0], [50.0, 50.0], [100.0, 50.0], [150.0, 50.0]], dtype=np.float32)
    ref_collinear = src_collinear.copy()
    mask = np.ones(4, dtype=bool)
    res = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    metrics = compute_registration_metrics(src_collinear, ref_collinear, mask, res, shape, runtime_seconds=0.05)
    assert metrics.inlier_count == 4
    assert metrics.rmse_px is not None
    # Collinear points have 0 convex hull area in 2D
    assert metrics.spatial_coverage_pct == 0.0


def test_outliers_excluded_from_error_metrics():
    """Verify that points with mask=False do NOT contaminate RMSE or median."""
    shape = (500, 500)
    inlier_src = np.full((20, 2), 100.0, dtype=np.float32)
    inlier_ref = np.full((20, 2), 100.0, dtype=np.float32)
    outlier_src = np.full((10, 2), 400.0, dtype=np.float32)
    outlier_ref = np.full((10, 2), 10.0, dtype=np.float32)

    src_pts = np.vstack([inlier_src, outlier_src])
    ref_pts = np.vstack([inlier_ref, outlier_ref])
    mask = np.array([True] * 20 + [False] * 10, dtype=bool)
    residuals = np.array([0.2] * 20 + [500.0] * 10, dtype=np.float32)

    metrics = compute_registration_metrics(
        source_points=src_pts,
        reference_points=ref_pts,
        inlier_mask=mask,
        residuals=residuals,
        image_shape=shape,
        tentative_match_count=50,
        geometric_inlier_count=35,
        spatially_balanced_count=30,
        subpixel_verified_count=20,
    )

    assert metrics.inlier_count == 20
    assert metrics.final_inlier_count == 20
    assert metrics.tentative_matches == 50
    assert metrics.geometric_inlier_count == 35
    assert metrics.spatially_balanced_count == 30
    assert metrics.subpixel_verified_count == 20
    assert metrics.rejected_match_count == 30
    assert pytest.approx(metrics.inlier_ratio_pct, 0.01) == 40.0
    assert pytest.approx(metrics.rmse_px, 0.001) == 0.2
    assert pytest.approx(metrics.median_residual_px, 0.001) == 0.2
    assert pytest.approx(metrics.p90_residual_px, 0.001) == 0.2
    assert pytest.approx(metrics.max_inlier_error_px, 0.001) == 0.2


def test_zero_inliers_handling():
    """Verify compute_registration_metrics handles zero inliers without crashing."""
    shape = (200, 200)
    src_pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
    ref_pts = np.array([[15, 15], [25, 25]], dtype=np.float32)
    mask = np.array([False, False], dtype=bool)
    residuals = np.array([50.0, 60.0], dtype=np.float32)

    metrics = compute_registration_metrics(
        source_points=src_pts,
        reference_points=ref_pts,
        inlier_mask=mask,
        residuals=residuals,
        image_shape=shape,
        tentative_match_count=10,
    )

    assert metrics.inlier_count == 0
    assert metrics.final_inlier_count == 0
    assert metrics.rmse_px is None
    assert metrics.median_residual_px is None
    assert metrics.p90_residual_px is None
    assert metrics.max_inlier_error_px is None
    assert metrics.inlier_ratio_pct == 0.0
    assert metrics.spatial_coverage_pct == 0.0
    assert metrics.spatial_uniformity_score == 0.0


def test_empty_points_handling():
    """Verify compute_registration_metrics handles completely empty arrays."""
    shape = (200, 200)
    src_pts = np.empty((0, 2), dtype=np.float32)
    ref_pts = np.empty((0, 2), dtype=np.float32)
    mask = np.empty((0,), dtype=bool)
    residuals = np.empty((0,), dtype=np.float32)

    metrics = compute_registration_metrics(
        source_points=src_pts,
        reference_points=ref_pts,
        inlier_mask=mask,
        residuals=residuals,
        image_shape=shape,
        tentative_match_count=0,
    )

    assert metrics.inlier_count == 0
    assert metrics.final_inlier_count == 0
    assert metrics.tentative_matches == 0
    assert metrics.rmse_px is None
    assert metrics.inlier_ratio_pct == 0.0


