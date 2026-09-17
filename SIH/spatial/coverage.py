"""
Spatial Coverage Computation for Planetary Correspondence Sets.
Measures the geometric footprint represented by validated correspondences.
"""
import numpy as np
from scipy.spatial import ConvexHull

def calculate_spatial_coverage(
    points: np.ndarray,
    image_shape: tuple[int, int],
) -> float:
    """
    Calculate the percentage of total image area spanned by the 2D convex hull of correspondences.
    Returns:
        coverage_pct: float in [0.0, 100.0]
    """
    if len(points) < 3:
        return 0.0

    h, w = image_shape[:2]
    total_area = float(h * w)
    if total_area <= 0:
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = float(hull.volume)  # In 2D, hull.volume represents the polygon area
        coverage_pct = (hull_area / total_area) * 100.0
        return round(float(np.clip(coverage_pct, 0.0, 100.0)), 2)
    except Exception:
        # Collinear points or degenerate hull
        return 0.0
