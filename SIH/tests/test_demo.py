"""Test suite for Phase 1B: Controlled lunar-like synthetic benchmark and scenarios."""
import numpy as np
import pytest
from demo.generator import generate_lunar_terrain, render_photometric_shading
from demo.scenarios import (
    generate_scenario_pair,
    list_available_scenarios,
    SCENARIOS_CATALOG,
    _apply_affine_to_points,
)

def test_dem_generator():
    """Verify procedural lunar terrain DEM generation and crater topography."""
    dem = generate_lunar_terrain(size=256, seed=42, crater_density="moderate")
    assert dem.shape == (256, 256)
    assert dem.dtype == np.float32
    # Ensure there is height variation (relief)
    assert float(np.ptp(dem)) > 20.0

def test_photometric_shading():
    """Verify directional illumination rendering produces valid 8-bit lunar radiance."""
    dem = generate_lunar_terrain(size=128, seed=42)
    rad_morning = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=25.0)
    rad_evening = render_photometric_shading(dem, sun_azimuth_deg=225.0, sun_elevation_deg=25.0)

    assert rad_morning.shape == (128, 128)
    assert rad_morning.dtype == np.uint8
    assert rad_evening.dtype == np.uint8
    # Shading must differ when solar azimuth flips by 180 degrees
    diff = np.abs(rad_morning.astype(float) - rad_evening.astype(float))
    assert np.mean(diff) > 10.0

def test_all_scenarios_generate():
    """Verify each of the 7 benchmark scenarios generates valid images and Ground Truth."""
    catalogs = list_available_scenarios()
    assert len(catalogs) == 7

    for scenario_id, title in SCENARIOS_CATALOG:
        data = generate_scenario_pair(scenario_id, size=256, seed=10)
        assert data.scenario_id == scenario_id
        assert data.source_image.width == 256
        assert data.reference_image.width == 256
        assert data.ground_truth_matrix.shape == (2, 3)
        assert len(data.sample_source_points) > 0
        assert len(data.sample_reference_points) == len(data.sample_source_points)

def test_ground_truth_point_mapping():
    """Verify sample points accurately follow model-specific transformation."""
    data = generate_scenario_pair("scenario_1_small_scale", size=256)
    src_pts = data.sample_source_points
    ref_pts = data.sample_reference_points
    re_mapped = _apply_affine_to_points(src_pts, data.ground_truth_matrix)

    # Verification: Ground truth mapping must have zero mathematical discrepancy
    error = np.linalg.norm(ref_pts - re_mapped, axis=1)
    assert np.max(error) < 1e-4

def test_scenario_5_illumination_identity():
    """Verify Scenario 5 has exact identity spatial transformation despite severe shadow shift."""
    data = generate_scenario_pair("scenario_5_illumination_shift", size=256)
    assert data.transformation_type == "identity"
    assert np.allclose(data.ground_truth_matrix, np.array([[1, 0, 0], [0, 1, 0]]))
    assert abs(data.sun_azimuth_ref - data.sun_azimuth_source) == 180.0
