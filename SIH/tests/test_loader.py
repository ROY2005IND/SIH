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


def test_rgb_lunar_image_uses_rgb_luminance_weights():
    """Ensure in-memory RGB imagery is not interpreted as OpenCV BGR data."""
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    lunar_image = load_lunar_image(rgb)

    grayscale = lunar_image.to_uint8()

    expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    assert np.array_equal(grayscale, expected)
    assert grayscale[0, 0] > grayscale[0, 1]


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
    assert pytest.approx(sun_vec[0], 0.01) == 0.707


def test_missing_gsd_returns_unknown():
    ohrc = OHRCAdapter().adapt({})
    assert ohrc["gsd"] == "UNKNOWN"
    assert ohrc["gsd_provenance"] == "UNKNOWN"
    assert ohrc["nominal_gsd"] == 0.25

    tmc2 = TMC2Adapter().adapt({})
    assert tmc2["gsd"] == "UNKNOWN"
    assert tmc2["gsd_provenance"] == "UNKNOWN"
    assert tmc2["nominal_gsd"] == 5.0


def test_sensor_detection_word_boundary_isolation():
    from metadata.adapters import SELENETCAdapter
    assert not isinstance(detect_sensor_adapter("weather_forecast_lunar.tif"), SELENETCAdapter)
    assert not isinstance(detect_sensor_adapter("crater_catch_orbit.tif"), SELENETCAdapter)
    assert not isinstance(detect_sensor_adapter("lunar_match_test.tif"), SELENETCAdapter)
    assert isinstance(detect_sensor_adapter("SELENE_TC_01.tif"), SELENETCAdapter)


def test_user_override_provenance(tmp_path):
    from metadata.reader import parse_lunar_metadata
    sample_img = tmp_path / "neutral_test_image.tif"
    tifffile.imwrite(sample_img, np.zeros((64, 64), dtype=np.uint8))
    meta = parse_lunar_metadata(sample_img, user_override={"gsd": 0.35, "instrument": "OHRC"})
    assert meta["gsd"] == 0.35
    assert meta["gsd_provenance"] == "USER_OVERRIDE"


def test_pds4_xml_parsing(tmp_path):
    from metadata.pds4_reader import parse_pds4_label
    pds4_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
        <Identification_Area>
            <logical_identifier>urn:isro:ch2:ohrc:ch2_ohr_n1_20200115</logical_identifier>
            <version_id>1.0</version_id>
            <title>Chandrayaan-2 OHRC Calibrated Radiance</title>
        </Identification_Area>
        <Observation_Area>
            <Time_Coordinates><start_date_time>2020-01-15T08:30:00.000Z</start_date_time></Time_Coordinates>
            <Investigation_Area><name>Chandrayaan-2</name></Investigation_Area>
            <Observing_System><Observing_System_Component><name>OHRC</name><type>Instrument</type></Observing_System_Component></Observing_System>
        </Observation_Area>
        <Discipline_Area>
            <Geometry>
                <solar_azimuth_angle>135.5</solar_azimuth_angle>
                <solar_elevation_angle>24.8</solar_elevation_angle>
                <incidence_angle>65.2</incidence_angle>
                <pixel_resolution>0.26</pixel_resolution>
            </Geometry>
        </Discipline_Area>
        <File_Area_Observational>
            <File><file_name>ch2_ohr_n1.tif</file_name></File>
            <Array_2D_Image>
                <Axis_Array><axis_name>Line</axis_name><elements>2048</elements><sequence_number>1</sequence_number></Axis_Array>
                <Axis_Array><axis_name>Sample</axis_name><elements>4096</elements><sequence_number>2</sequence_number></Axis_Array>
            </Array_2D_Image>
        </File_Area_Observational>
    </Product_Observational>"""
    xml_path = tmp_path / "test_pds4.xml"
    xml_path.write_text(pds4_xml_content, encoding="utf-8")

    meta = parse_pds4_label(xml_path)
    assert meta is not None
    assert meta.product_id == "ch2_ohr_n1_20200115"
    assert meta.sun_azimuth == 135.5
    assert meta.sun_elevation == 24.8
    assert meta.gsd == 0.26
    assert meta.phase_angle is None
