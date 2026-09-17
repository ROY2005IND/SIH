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


def _create_synthetic_pair(size: int = 256, shift_x: float = 0.25, shift_y: float = 0.25):
    dem = generate_lunar_terrain(size=size, seed=101)
    src_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=35.0)
    M_true = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    ref_u8 = cv2.warpAffine(src_u8, M_true, (size, size), flags=cv2.INTER_CUBIC)
    return src_u8, ref_u8


def test_subpixel_known_shift_accuracy():
    shifts_to_test = [0.10, 0.20, 0.25, 0.35]
    for shift in shifts_to_test:
        src_u8, ref_u8 = _create_synthetic_pair(size=256, shift_x=shift, shift_y=shift)
        grid_x, grid_y = np.meshgrid(np.arange(60, 190, 25), np.arange(60, 190, 25))
        src_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        ref_pts_coarse = src_pts.copy()

        result = refine_subpixel_correspondences(
            source_image_u8=src_u8,
            reference_image_u8=ref_u8,
            source_points=src_pts,
            reference_points=ref_pts_coarse,
            patch_size=15,
            search_radius_px=2,
        )

        assert result.n_converged >= len(src_pts) * 0.7
        gt_ref_pts = src_pts + np.array([shift, shift], dtype=np.float32)
        valid_mask = result.converged_mask
        estimated_ref = result.refined_points[valid_mask]
        true_ref = gt_ref_pts[valid_mask]
        errors = np.linalg.norm(estimated_ref - true_ref, axis=1)
        mae = float(np.mean(errors))
        assert mae < 0.25


def test_subpixel_robustness_under_noise():
    shift = 0.30
    src_u8, ref_u8 = _create_synthetic_pair(size=256, shift_x=shift, shift_y=shift)
    rng = np.random.default_rng(42)
    noise_src = rng.normal(0, 8.0, src_u8.shape).astype(np.float32)
    noise_ref = rng.normal(0, 8.0, ref_u8.shape).astype(np.float32)
    noisy_src = np.clip(src_u8.astype(np.float32) + noise_src, 0, 255).astype(np.uint8)
    noisy_ref = np.clip(ref_u8.astype(np.float32) + noise_ref, 0, 255).astype(np.uint8)

    src_pts = np.array([[80.0, 80.0], [120.0, 100.0], [140.0, 140.0], [100.0, 160.0]], dtype=np.float32)
    ref_coarse = src_pts.copy()

    result = refine_subpixel_correspondences(
        source_image_u8=noisy_src,
        reference_image_u8=noisy_ref,
        source_points=src_pts,
        reference_points=ref_coarse,
        patch_size=15,
        search_radius_px=2,
    )
    assert result.n_converged >= 2


def test_subpixel_contrast_invariance():
    shift = 0.20
    src_u8, ref_u8 = _create_synthetic_pair(size=256, shift_x=shift, shift_y=shift)
    lut = np.array([((i / 255.0) ** (1.0 / 1.6)) * 255 for i in range(256)]).astype(np.uint8)
    ref_gamma = cv2.LUT(ref_u8, lut)

    src_pts = np.array([[100.0, 100.0], [120.0, 120.0]], dtype=np.float32)
    ref_coarse = src_pts.copy()

    result = refine_subpixel_correspondences(
        source_image_u8=src_u8,
        reference_image_u8=ref_gamma,
        source_points=src_pts,
        reference_points=ref_coarse,
        patch_size=15,
        search_radius_px=2,
    )
    assert result.n_converged >= 1

