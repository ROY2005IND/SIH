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
    """Quantitative evaluation metrics for a single registration run.

    Stage-level point counts track the match population through every
    pipeline stage so that no count is conflated with another:

        tentative → geometric_inliers → spatially_balanced → subpixel_verified → final_inliers
    """
    tentative_matches: int
    geometric_inlier_count: int
    spatially_balanced_count: int
    subpixel_verified_count: int
    final_inlier_count: int
    rejected_match_count: int

    # Legacy alias — kept for backward compatibility with export/tracker/CLI.
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


def _empty_metrics(
    tentative_matches: int = 0,
    runtime_seconds: float = 0.0,
) -> RegistrationMetrics:
    """Return a zero-valued metrics object for failed or empty pipelines."""
    return RegistrationMetrics(
        tentative_matches=tentative_matches,
        geometric_inlier_count=0,
        spatially_balanced_count=0,
        subpixel_verified_count=0,
        final_inlier_count=0,
        rejected_match_count=tentative_matches,
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


def compute_registration_metrics(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    inlier_mask: np.ndarray | None,
    residuals: np.ndarray | None,
    image_shape: tuple[int, int],
    runtime_seconds: float = 0.0,
    grid_size: int = 6,
    *,
    tentative_match_count: int | None = None,
    geometric_inlier_count: int | None = None,
    spatially_balanced_count: int | None = None,
    subpixel_verified_count: int | None = None,
) -> RegistrationMetrics:
    """Compute rigorous registration and correspondence metrics.

    Parameters
    ----------
    source_points, reference_points : (N, 2) float32
        The *final verified* point arrays to compute metrics over.
    inlier_mask : (N,) bool or None
        Must match the length of source_points/reference_points.
    residuals : (N,) float32 or None
        Transfer residuals for the same N points.
    tentative_match_count : int, optional
        Original tentative match count (before any filtering).
        Defaults to len(source_points) if not provided.
    geometric_inlier_count, spatially_balanced_count, subpixel_verified_count : int, optional
        Stage-level counts.  Defaults to 0 if not provided.
    """
    n_pts = len(source_points)
    n_tentative = tentative_match_count if tentative_match_count is not None else n_pts

    if n_pts == 0 or inlier_mask is None or residuals is None:
        return _empty_metrics(tentative_matches=n_tentative, runtime_seconds=runtime_seconds)

    # Validate mask/residual length consistency
    if len(inlier_mask) != n_pts:
        raise ValueError(
            f"Inlier mask length ({len(inlier_mask)}) does not match point count ({n_pts}). "
            f"This indicates a pipeline bug where a mask from an earlier stage is being "
            f"applied to a different point population."
        )
    if len(residuals) != n_pts:
        raise ValueError(
            f"Residuals length ({len(residuals)}) does not match point count ({n_pts})."
        )

    inlier_pts_ref = reference_points[inlier_mask]
    inlier_res = residuals[inlier_mask]
    n_inliers = int(np.sum(inlier_mask))

    # Scientific validation requirement: A 2D planar transformation (Similarity, Affine, Homography)
    # requires at least 4 independent correspondences. When n_inliers < 4, residual error metrics
    # have no independent geometric validation meaning and reporting a low/zero RMSE is scientifically misleading.
    if n_inliers >= 4 and len(inlier_res) >= 4:
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

    # Compute the inlier ratio against *tentative* matches, not just the
    # current working set — this gives an honest overall inlier proportion.
    inlier_ratio = round((n_inliers / max(n_tentative, 1)) * 100.0, 2)

    geo_inliers = geometric_inlier_count if geometric_inlier_count is not None else 0
    spat_balanced = spatially_balanced_count if spatially_balanced_count is not None else 0
    subpx_verified = subpixel_verified_count if subpixel_verified_count is not None else 0
    rejected = n_tentative - n_inliers

    return RegistrationMetrics(
        tentative_matches=n_tentative,
        geometric_inlier_count=geo_inliers,
        spatially_balanced_count=spat_balanced,
        subpixel_verified_count=subpx_verified,
        final_inlier_count=n_inliers,
        rejected_match_count=rejected,
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
