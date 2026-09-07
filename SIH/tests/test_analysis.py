"""Test suite for Phase 2: Pre-matching Image Quality & Tiered Overlap Analysis."""
import numpy as np
import pytest
from preprocessing.loader import LunarImage
from analysis.image_quality import analyze_image_quality
from analysis.overlap import estimate_overlap
from demo.generator import generate_lunar_terrain, render_photometric_shading

def test_image_quality_high_relief():
    """Verify quality analysis on cratered lunar terrain returns rich texture and optimal readiness."""
    dem = generate_lunar_terrain(size=256, seed=42, crater_density="high")
    rad = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=25.0)

    lunar = LunarImage(
        image=rad,
        width=256,
        height=256,
        channels=1,
        mission="Chandrayaan-2",
        instrument="OHRC",
    )

    report = analyze_image_quality(lunar)
    assert report.valid_pixel_pct == 100.0
    assert report.texture_entropy > 4.5
    assert report.texture_level in ["RICH", "MODERATE"]
    assert report.contrast_level in ["HIGH", "MODERATE"]
    assert report.readiness_status in ["OPTIMAL", "ACCEPTABLE"]

def test_image_quality_flat_low_contrast():
    """Verify quality analysis flags low contrast and low texture on uniform surface."""
    flat_arr = np.full((128, 128), 128, dtype=np.uint8)
    flat_lunar = LunarImage(
        image=flat_arr,
        width=128,
        height=128,
        channels=1,
    )

    report = analyze_image_quality(flat_lunar)
    assert report.contrast_level == "LOW"
    assert report.texture_level == "LOW"
    assert "WARNING" in report.readiness_status

def test_overlap_level1_metadata():
    """Verify Level 1 metadata-assisted overlap calculation using bounding boxes."""
    dummy_arr = np.zeros((100, 100), dtype=np.uint8)
    src = LunarImage(
        image=dummy_arr,
        width=100,
        height=100,
        channels=1,
        gsd=0.25,
        auxiliary={"bbox": [10.0, 10.0, 20.0, 20.0]},  # 10x10 area = 100
    )
    ref = LunarImage(
        image=dummy_arr,
        width=100,
        height=100,
        channels=1,
        gsd=1.0,
        auxiliary={"bbox": [15.0, 10.0, 25.0, 20.0]},  # 5x10 overlap = 50
    )

    report = estimate_overlap(src, ref)
    assert report.level == "Level 1: Metadata-assisted"
    assert report.confidence == "HIGH"
    assert report.overlap_pct == 50.0
    assert report.estimated_scale_ratio == 4.0

def test_overlap_level2_image_based():
    """Verify Level 2 image-based approximate overlap on translated image pair."""
    dem = generate_lunar_terrain(size=256, seed=42)
    rad = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)

    # Shift by 30 pixels horizontally
    rad_shifted = np.roll(rad, shift=30, axis=1)

    src = LunarImage(image=rad, width=256, height=256, channels=1)
    ref = LunarImage(image=rad_shifted, width=256, height=256, channels=1)

    report = estimate_overlap(src, ref)
    assert "Level 2" in report.level
    assert report.overlap_pct is not None
    assert report.overlap_pct > 60.0  # Approx (256-30)/256 ~ 88%

def test_overlap_level3_uncorrelated():
    """Verify Level 3 fallback when images are completely uncorrelated random noise."""
    rng = np.random.default_rng(123)
    n1 = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    n2 = rng.integers(0, 255, (128, 128), dtype=np.uint8)

    src = LunarImage(image=n1, width=128, height=128, channels=1)
    ref = LunarImage(image=n2, width=128, height=128, channels=1)

    report = estimate_overlap(src, ref)
    # Correlation between random uncorrelated noise must fail below threshold
    assert "Level 3" in report.level
    assert report.confidence == "UNKNOWN"
