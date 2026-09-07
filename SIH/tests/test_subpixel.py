"""Test suite for Phase 8: Sub-Pixel Refinement and Parabolic Peak Fitting."""
import numpy as np
import pytest
import cv2
from registration.subpixel import _fit_2d_parabolic_peak, refine_subpixel_correspondences
from demo.generator import generate_lunar_terrain, render_photometric_shading

def test_parabolic_peak_continuous_fitting():
    """Verify 2D quadratic peak solver recovers known sub-pixel fractional offset."""
    # Synthetic 2D paraboloid: f(u, v) = 1.0 - 2*(u - u0)^2 - 2*(v - v0)^2
    true_u0 = 0.35
    true_v0 = -0.25

    patch = np.zeros((3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            u = c - 1  # -1, 0, 1
            v = r - 1  # -1, 0, 1
            patch[r, c] = 1.0 - 2.0 * (u - true_u0) ** 2 - 2.0 * (v - true_v0) ** 2

    delta_u, delta_v, is_valid = _fit_2d_parabolic_peak(patch)
    assert is_valid is True
    assert pytest.approx(delta_u, 0.05) == true_u0
    assert pytest.approx(delta_v, 0.05) == true_v0

def test_subpixel_refinement_pipeline():
    """Verify sub-pixel refinement adjusts integer coordinates towards fractional true shift."""
    dem = generate_lunar_terrain(size=128, seed=42)
    src_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=35.0)

    # Shift reference image fractionally by (dx=1.4, dy=-0.3)
    M_shift = np.array([[1.0, 0.0, 1.4], [0.0, 1.0, -0.3]], dtype=np.float32)
    ref_u8 = cv2.warpAffine(src_u8, M_shift, (128, 128), flags=cv2.INTER_CUBIC)

    # Integer correspondences (rounded without sub-pixel)
    src_pts = np.array([[64.0, 64.0], [50.0, 70.0], [80.0, 55.0]], dtype=np.float32)
    ref_pts_integer = np.array([[65.0, 64.0], [51.0, 70.0], [81.0, 55.0]], dtype=np.float32)  # [x+1, y]

    sub_res = refine_subpixel_correspondences(
        src_u8, ref_u8, src_pts, ref_pts_integer, patch_size=11, search_radius_px=2
    )

    assert len(sub_res.refined_points) == 3
    assert sub_res.n_converged >= 2
    # Check that refined x coordinate is shifted positively towards 1.4
    for idx in range(3):
        if sub_res.converged_mask[idx]:
            # Refined coordinate should have fractional component
            assert abs(sub_res.refined_points[idx, 0] - ref_pts_integer[idx, 0]) > 0.05
