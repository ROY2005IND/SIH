"""Test suite for Phase 7: Adaptive Spatial Match Balancer and Coverage."""
import numpy as np
import pytest
from spatial.coverage import calculate_spatial_coverage
from spatial.distribution import compute_spatial_uniformity, AdaptiveSpatialBalancer

def test_spatial_coverage():
    """Verify 2D convex hull spatial coverage percentage calculation."""
    shape = (100, 100)
    # Square covering 50x50 = 2500 px out of 10000 px = 25%
    pts = np.array([[25.0, 25.0], [75.0, 25.0], [75.0, 75.0], [25.0, 75.0]], dtype=np.float32)
    cov = calculate_spatial_coverage(pts, shape)
    assert pytest.approx(cov, 0.5) == 25.0

    # Degenerate points (<3 points)
    assert calculate_spatial_coverage(np.array([[10.0, 10.0]]), shape) == 0.0

def test_spatial_uniformity_score():
    """Verify uniformity score drops when points cluster in a single corner."""
    shape = (200, 200)
    # 1. Perfectly uniform grid of points
    grid_y, grid_x = np.mgrid[20:180:6j, 20:180:6j]
    uniform_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    score_unif, occ_unif, _ = compute_spatial_uniformity(uniform_pts, shape, grid_size=6)
    assert score_unif > 70.0
    assert occ_unif > 80.0

    # 2. Clustered points in one single cell (e.g. on one crater rim)
    clustered_pts = np.random.uniform(5.0, 25.0, (36, 2)).astype(np.float32)
    score_clust, occ_clust, _ = compute_spatial_uniformity(clustered_pts, shape, grid_size=6)
    assert score_clust < 25.0
    assert occ_clust < 20.0

def test_adaptive_spatial_balancer():
    """Verify balancer prunes dense clusters and bounds points per cell."""
    shape = (200, 200)
    # Create an extremely skewed cluster of 100 points in cell (0, 0) and 5 in other cells
    rng = np.random.default_rng(42)
    c1 = rng.uniform(5.0, 25.0, (100, 2)).astype(np.float32)
    c2 = rng.uniform(50.0, 150.0, (20, 2)).astype(np.float32)
    raw_src = np.vstack([c1, c2])
    raw_ref = raw_src.copy()
    confs = rng.uniform(0.6, 0.95, len(raw_src)).astype(np.float32)

    balancer = AdaptiveSpatialBalancer(grid_size=6, max_matches_per_cell=10)
    result = balancer.balance(raw_src, raw_ref, confs, shape)

    # Max points in any cell must not exceed max_matches_per_cell
    assert np.max(result.cell_counts_matrix) <= 10
    # Balanced count must be significantly lower than raw skewed count
    assert len(result.balanced_source_points) < len(raw_src)
    # Balanced uniformity must be higher than raw skewed uniformity
    assert result.balanced_uniformity_score >= result.raw_uniformity_score
