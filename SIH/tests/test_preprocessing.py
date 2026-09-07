"""Test suite for Phase 3: Preprocessing engine (illumination, PCA, pyramid, tiling)."""
import numpy as np
import pytest
from preprocessing.normalization import percentile_stretch, match_radiometric_histogram
from preprocessing.illumination import (
    apply_clahe,
    apply_gradient_magnitude,
    apply_edge_representation,
    apply_phase_congruency,
    apply_preprocessing,
)
from preprocessing.hyperspectral import HyperspectralProcessor
from preprocessing.pyramid import ScalePyramid
from preprocessing.tiling import RasterTiler

def test_normalization_and_histogram_matching():
    """Verify percentile stretch and histogram matching."""
    arr = np.linspace(50.0, 500.0, 10000).reshape((100, 100))
    stretched = percentile_stretch(arr)
    assert stretched.dtype == np.uint8
    assert stretched.min() == 0
    assert stretched.max() == 255

    src_dark = (np.random.rand(64, 64) * 80).astype(np.uint8)
    ref_bright = (np.random.rand(64, 64) * 200 + 55).astype(np.uint8)
    matched = match_radiometric_histogram(src_dark, ref_bright)
    assert matched.mean() > src_dark.mean()

def test_illumination_representations():
    """Verify CLAHE, Gradient, Edge, and Phase representations."""
    u8 = (np.random.rand(128, 128) * 255).astype(np.uint8)

    clahe = apply_clahe(u8)
    assert clahe.shape == (128, 128)
    assert clahe.dtype == np.uint8

    grad = apply_gradient_magnitude(u8)
    assert grad.shape == (128, 128)
    assert grad.dtype == np.uint8

    edge = apply_edge_representation(u8)
    assert edge.shape == (128, 128)
    assert edge.dtype == np.uint8

    phase = apply_phase_congruency(u8)
    assert phase.shape == (128, 128)
    assert phase.dtype == np.uint8

    for mode in ["original", "clahe", "gradient", "edge", "phase"]:
        out = apply_preprocessing(u8, method=mode)
        assert out.shape == (128, 128)

def test_hyperspectral_processor():
    """Verify IIRS 3D cube band extraction and PCA reduction."""
    cube = np.random.rand(64, 64, 25).astype(np.float32)
    proc = HyperspectralProcessor(cube)
    assert proc.n_bands == 25

    b5 = proc.extract_band(5)
    assert b5.shape == (64, 64)
    assert b5.dtype == np.uint8

    b_wave, idx, um = proc.extract_band_by_wavelength(1.5)
    assert b_wave.shape == (64, 64)
    assert idx >= 0

    pc1 = proc.get_principal_component(0)
    assert pc1.shape == (64, 64)
    assert pc1.dtype == np.uint8

def test_scale_pyramid():
    """Verify Gaussian scale pyramid coordinate mappings."""
    base = np.zeros((256, 256), dtype=np.uint8)
    pyr = ScalePyramid(base, n_levels=3)
    assert len(pyr.levels) == 3
    assert pyr.levels[0].width == 256
    assert pyr.levels[1].width == 128
    assert pyr.levels[2].width == 64

    # Point at level 1 (x=30, y=40) should map to (x=60, y=80) at level 0
    pts_lvl1 = np.array([[30.0, 40.0]], dtype=np.float32)
    pts_native = pyr.to_native_coordinates(pts_lvl1, level=1)
    assert pts_native[0, 0] == 60.0
    assert pts_native[0, 1] == 80.0

def test_raster_tiling():
    """Verify spatial tiling and coordinate offsetting."""
    big_img = np.zeros((500, 500), dtype=np.uint8)
    tiler = RasterTiler(big_img, tile_size=256, overlap_px=64)
    tiles = list(tiler.generate_tiles())
    assert len(tiles) >= 4

    tile0 = tiles[0]
    local_pt = np.array([[10.0, 15.0]], dtype=np.float32)
    global_pt = RasterTiler.map_tile_points_to_global(local_pt, tile0)
    assert global_pt[0, 0] == 10.0 + tile0.global_x_offset
    assert global_pt[0, 1] == 15.0 + tile0.global_y_offset
