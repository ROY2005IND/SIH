"""
Explainable Lunar Registration Quality Score Engine.
Computes an internal score in range [0, 100].
NOTE: Clearly labeled as an internal experimental score — not an official ISRO metric.
"""
from dataclasses import dataclass
import numpy as np
from .metrics import RegistrationMetrics

@dataclass
class LunarRegistrationQualityScore:
    """Internal experimental composite quality score with transparent component sub-scores."""
    total_score: int              # [0, 100]
    inlier_score: float           # [0, 30]
    coverage_score: float         # [0, 25]
    uniformity_score: float       # [0, 25]
    residual_score: float         # [0, 20]
    rating_label: str             # "EXEMPLARY", "RELIABLE", "ACCEPTABLE", "MARGINAL", "UNRELIABLE"
    disclaimer: str = "Internal experimental score — not an official ISRO metric"

def calculate_quality_score(metrics: RegistrationMetrics) -> LunarRegistrationQualityScore:
    """
    Calculate normalized explainable quality score based on measurable scientific metrics:
    - Inlier Ratio (0–30 pts): 80%+ gives 30 pts, drops below 30%.
    - Spatial Coverage (0–25 pts): 60%+ coverage gives 25 pts.
    - Spatial Uniformity (0–25 pts): Based on normalized grid entropy.
    - Residual Precision (0–20 pts): RMSE < 0.5px gives 20 pts, 1.0px gives 16 pts, drops towards 3.0px.
    """
    if metrics.inlier_count < 4 or metrics.rmse_px is None:
        return LunarRegistrationQualityScore(
            total_score=0,
            inlier_score=0.0,
            coverage_score=0.0,
            uniformity_score=0.0,
            residual_score=0.0,
            rating_label="UNRELIABLE",
        )

    # 1. Inlier ratio score (Max 30)
    ratio_frac = np.clip(metrics.inlier_ratio_pct / 80.0, 0.0, 1.0)
    inlier_pts = float(30.0 * ratio_frac)

    # 2. Spatial coverage score (Max 25)
    cov_frac = np.clip(metrics.spatial_coverage_pct / 60.0, 0.0, 1.0)
    coverage_pts = float(25.0 * cov_frac)

    # 3. Spatial uniformity score (Max 25)
    unif_frac = np.clip(metrics.spatial_uniformity_score / 80.0, 0.0, 1.0)
    uniformity_pts = float(25.0 * unif_frac)

    # 4. Residual error score (Max 20)
    # RMSE <= 0.5px -> 20, RMSE = 1.0px -> 16, RMSE = 3.0px -> 4, RMSE > 4.0px -> 0
    rmse = metrics.rmse_px
    res_frac = np.clip((3.5 - rmse) / 3.0, 0.0, 1.0)
    residual_pts = float(20.0 * res_frac)

    total = int(round(inlier_pts + coverage_pts + uniformity_pts + residual_pts))
    total = max(0, min(100, total))

    if total >= 85:
        label = "EXEMPLARY"
    elif total >= 70:
        label = "RELIABLE"
    elif total >= 55:
        label = "ACCEPTABLE"
    elif total >= 40:
        label = "MARGINAL"
    else:
        label = "UNRELIABLE"

    return LunarRegistrationQualityScore(
        total_score=total,
        inlier_score=round(inlier_pts, 1),
        coverage_score=round(coverage_pts, 1),
        uniformity_score=round(uniformity_pts, 1),
        residual_score=round(residual_pts, 1),
        rating_label=label,
    )
