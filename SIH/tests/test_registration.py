"""Test suite for Phase 4: Classical registration components (SIFT, RANSAC, model selection, warping)."""
import numpy as np
import pytest
from features.classical import extract_sift_features, extract_akaze_features
from matching.classical import ClassicalMatcher
from registration.transformation import apply_transformation, invert_transformation, calculate_residuals
from registration.ransac import estimate_robust_transformation
from registration.model_selection import select_best_transformation_model
from registration.warping import warp_image_to_reference, compute_difference_map, create_alpha_overlay
from demo.generator import generate_lunar_terrain, render_photometric_shading

def test_classical_feature_extractors():
    """Verify SIFT and AKAZE keypoint and descriptor extraction."""
    dem = generate_lunar_terrain(size=256, seed=42)
    rad = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)

    # SIFT
    pts_sift, desc_sift, _ = extract_sift_features(rad)
    assert len(pts_sift) > 50
    assert desc_sift is not None
    assert desc_sift.shape[1] == 128

    # AKAZE
    pts_akaze, desc_akaze, _ = extract_akaze_features(rad)
    assert len(pts_akaze) > 30
    assert desc_akaze is not None

def test_classical_matcher_on_translated_pair():
    """Verify SIFT descriptor matching on translated synthetic image."""
    dem = generate_lunar_terrain(size=256, seed=42)
    src_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)
    # Translate by 15 pixels
    ref_u8 = np.roll(src_u8, shift=15, axis=1)

    matcher = ClassicalMatcher(method="sift")
    src_pts, ref_pts, confs, _ = matcher.find_matches(src_u8, ref_u8)
    assert len(src_pts) > 10
    assert len(src_pts) == len(ref_pts) == len(confs)

def test_robust_transformation_estimation():
    """Verify RANSAC accurately recovers known affine transformation."""
    rng = np.random.default_rng(42)
    src_pts = rng.uniform(20.0, 200.0, (50, 2)).astype(np.float32)

    # True affine matrix: scale 1.1, rotation ~5 deg, translation (10, -5)
    true_M = np.array([[1.09, -0.09, 10.0], [0.09, 1.09, -5.0]], dtype=np.float64)
    ref_pts = apply_transformation(src_pts, true_M)

    # Add 10 severe outliers
    outliers = rng.uniform(0.0, 250.0, (10, 2)).astype(np.float32)
    src_noisy = np.vstack([src_pts, rng.uniform(0.0, 250.0, (10, 2)).astype(np.float32)])
    ref_noisy = np.vstack([ref_pts, outliers])

    M_est, mask, residuals = estimate_robust_transformation(src_noisy, ref_noisy, model="affine")
    assert M_est is not None
    assert mask is not None
    assert np.sum(mask) >= 45  # Correctly identifies almost all 50 inliers
    assert np.allclose(M_est, true_M, atol=0.1)

def test_model_selection_engine():
    """Verify model selection picks Similarity or Affine and produces valid rationale."""
    rng = np.random.default_rng(42)
    src_pts = rng.uniform(30.0, 180.0, (40, 2)).astype(np.float32)
    true_M = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0]], dtype=np.float64)
    ref_pts = apply_transformation(src_pts, true_M)

    model, M, mask, res, reason, evals = select_best_transformation_model(src_pts, ref_pts)
    assert model in ["similarity", "affine", "homography"]
    assert "Selected" in reason
    assert len(evals) >= 1

def test_warping_and_difference():
    """Verify image warping and difference map generation."""
    img = np.full((100, 100), 120, dtype=np.uint8)
    identity_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    warped = warp_image_to_reference(img, identity_M, (100, 100))
    assert warped.shape == (100, 100)
    assert np.array_equal(warped, img)

    diff, mean_diff = compute_difference_map(warped, img)
    assert mean_diff == 0.0

    overlay = create_alpha_overlay(warped, img)
    assert overlay.shape == (100, 100, 3)
