"""
Comprehensive Scientific Evaluation Metrics for Lunar Correspondence & Registration.
Never fabricates metrics; all values are computed mathematically from correspondence data.
"""
from dataclasses import dataclass
import numpy as np
from spatial.coverage import calculate_spatial_coverage
from spatial.distribution import compute_spatial_uniformity

@dataclass
class RegistrationMetrics:
    """Quantitative evaluation metrics for a single registration run."""
    tentative_matches: int
    inlier_count: int
    inlier_ratio_pct: float
    rmse_px: float | None
    median_residual_px: float | None
    p90_residual_px: float | None
    max_inlier_error_px: float | None
    spatial_coverage_pct: float
    spatial_uniformity_score: float
    grid_occupancy_pct: float
    runtime_seconds: float

def compute_registration_metrics(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    inlier_mask: np.ndarray | None,
    residuals: np.ndarray | None,
    image_shape: tuple[int, int],
    runtime_seconds: float = 0.0,
    grid_size: int = 6,
) -> RegistrationMetrics:
    """Compute rigorous registration and correspondence metrics."""
    n_pts = len(source_points)
    if n_pts == 0 or inlier_mask is None or residuals is None:
        return RegistrationMetrics(
            tentative_matches=n_pts,
            inlier_count=0,
            inlier_ratio_pct=0.0,
            rmse_px=None,
            median_residual_px=None,
            p90_residual_px=None,
            max_inlier_error_px=None,
            spatial_coverage_pct=0.0,
            spatial_uniformity_score=0.0,
            grid_occupancy_pct=0.0,
            runtime_seconds=runtime_seconds,
        )

    inlier_pts_ref = reference_points[inlier_mask]
    inlier_res = residuals[inlier_mask]
    n_inliers = int(np.sum(inlier_mask))
    inlier_ratio = round((n_inliers / max(n_pts, 1)) * 100.0, 2)

    if len(inlier_res) > 0:
        rmse = round(float(np.sqrt(np.mean(inlier_res ** 2))), 3)
        med_res = round(float(np.median(inlier_res)), 3)
        p90_res = round(float(np.percentile(inlier_res, 90)), 3)
        max_err = round(float(np.max(inlier_res)), 3)
    else:
        rmse, med_res, p90_res, max_err = None, None, None, None

    # Spatial coverage of inliers
    coverage_pct = calculate_spatial_coverage(inlier_pts_ref, image_shape)

    # Spatial uniformity of inliers
    uniformity, occupancy, _ = compute_spatial_uniformity(inlier_pts_ref, image_shape, grid_size=grid_size)

    return RegistrationMetrics(
        tentative_matches=n_pts,
        inlier_count=n_inliers,
        inlier_ratio_pct=inlier_ratio,
        rmse_px=rmse,
        median_residual_px=med_res,
        p90_residual_px=p90_res,
        max_inlier_error_px=max_err,
        spatial_coverage_pct=coverage_pct,
        spatial_uniformity_score=uniformity,
        grid_occupancy_pct=occupancy,
        runtime_seconds=round(runtime_seconds, 3),
    )
