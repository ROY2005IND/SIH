"""Test suite for Phase 1: Data engine, LunarImage loader, and metadata adapters."""
import numpy as np
import pytest
import tifffile
import cv2
from preprocessing.loader import LunarImage, load_lunar_image
from metadata.adapters import OHRCAdapter, TMC2Adapter, IIRSAdapter, LRONACAdapter, detect_sensor_adapter
from metadata.geometry import calculate_gsd_scale_ratio, sun_vector_from_angles

def test_lunar_image_dataclass():
    """Verify LunarImage properties, uint8 conversion, and metadata export."""
    synthetic_arr = np.linspace(100, 5000, 10000, dtype=np.uint16).reshape((100, 100))
    img = LunarImage(
        image=synthetic_arr,
        width=100,
        height=100,
        channels=1,
        mission="Chandrayaan-2",
        instrument="OHRC",
        gsd=0.25,
    )
    assert img.width == 100
    assert img.height == 100
    assert img.channels == 1
    assert img.mission == "Chandrayaan-2"
    assert img.sun_azimuth == "Not available"  # Honest missing value

    u8 = img.to_uint8()
    assert u8.dtype == np.uint8
    assert u8.shape == (100, 100)
    assert u8.min() >= 0
    assert u8.max() <= 255

def test_sensor_adapters():
    """Verify mission adapters extract known values and leave missing fields as 'Not available'."""
    ohrc_raw = {"gsd": 0.25, "sun_azimuth": 45.0}
    ohrc = OHRCAdapter().adapt(ohrc_raw)
    assert ohrc["mission"] == "Chandrayaan-2"
    assert ohrc["instrument"] == "OHRC"
    assert ohrc["gsd"] == 0.25
    assert ohrc["sun_azimuth"] == 45.0
    assert ohrc["sun_elevation"] == "Not available"

    iirs = IIRSAdapter().adapt({})
    assert iirs["instrument"] == "IIRS"
    assert "250" in iirs["wavelength"]

    nac = LRONACAdapter().adapt({"gsd": 0.5})
    assert nac["mission"] == "Lunar Reconnaissance Orbiter (LRO)"
    assert nac["instrument"] == "LROC NAC"

def test_detect_sensor_adapter():
    """Verify auto-detection of sensor adapter from filenames."""
    assert isinstance(detect_sensor_adapter("ch2_ohrc_ref_01.tif"), OHRCAdapter)
    assert isinstance(detect_sensor_adapter("ch2_tmc2_stereo.tif"), TMC2Adapter)
    assert isinstance(detect_sensor_adapter("lro_nac_m11234.tif"), LRONACAdapter)

def test_file_loader(tmp_path):
    """Verify load_lunar_image reads GeoTIFF and PNG files correctly."""
    # Write synthetic GeoTIFF
    test_arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
    tif_path = tmp_path / "ch2_ohrc_sample.tif"
    tifffile.imwrite(tif_path, test_arr)

    lunar_img = load_lunar_image(tif_path)
    assert lunar_img.width == 128
    assert lunar_img.height == 128
    assert lunar_img.instrument == "OHRC"

    # Write PNG
    png_path = tmp_path / "test_image.png"
    cv2.imwrite(str(png_path), test_arr)
    lunar_png = load_lunar_image(png_path)
    assert lunar_png.width == 128
    assert lunar_png.height == 128

def test_geometry_calculations():
    """Verify solar vector and GSD scale ratio calculations."""
    scale_ratio = calculate_gsd_scale_ratio(source_gsd=0.25, ref_gsd=5.0)
    assert scale_ratio == 20.0

    sun_vec = sun_vector_from_angles(azimuth_deg=90.0, elevation_deg=45.0)
    assert sun_vec is not None
    assert pytest.approx(np.linalg.norm(sun_vec), 0.001) == 1.0
    assert pytest.approx(sun_vec[0], 0.01) == 0.707  # East component
