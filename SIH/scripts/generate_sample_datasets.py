"""
Utility script to generate sample lunar imagery pairs in data/samples/
for local file testing and offline ISRO evaluation demonstrations.
Includes metadata sidecars and standard GeoTIFF/TIFF files.
"""
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tifffile
import cv2

from demo.generator import generate_lunar_terrain, render_photometric_shading
from utils.paths import SAMPLES_DIR

def generate_samples(size: int = 512, seed: int = 42):
    """Generate standardized sample datasets for offline testing."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating sample datasets into: {SAMPLES_DIR}")

    # 1. OHRC vs TMC-2 Cross-Scale Pair
    dem_cratered = generate_lunar_terrain(size=size, seed=seed, crater_density="moderate")
    # Base terrain rendered at 45 deg sun azimuth
    base_tmc = render_photometric_shading(dem_cratered, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)

    # OHRC simulation: 1.5x zoom + 12 px shift
    center = (size / 2.0, size / 2.0)
    M_scale = cv2.getRotationMatrix2D(center, 0.0, 1.3)
    M_scale[0, 2] += 15.0
    M_scale[1, 2] -= 10.0
    ohrc_img = cv2.warpAffine(base_tmc, M_scale, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    tifffile.imwrite(SAMPLES_DIR / "sample_ohrc.tif", ohrc_img)
    with open(SAMPLES_DIR / "sample_ohrc.json", "w", encoding="utf-8") as f:
        json.dump({
            "mission": "Chandrayaan-2",
            "instrument": "OHRC",
            "sensor": "Optical High Resolution Camera",
            "gsd": 0.25,
            "sun_azimuth": 45.0,
            "sun_elevation": 30.0,
            "incidence_angle": 60.0,
            "emission_angle": 0.0,
            "phase_angle": 60.0,
            "wavelength": "Panchromatic (450-900 nm)",
            "description": "Simulated Chandrayaan-2 OHRC high-resolution observation",
        }, f, indent=2)

    tifffile.imwrite(SAMPLES_DIR / "sample_tmc2.tif", base_tmc)
    with open(SAMPLES_DIR / "sample_tmc2.json", "w", encoding="utf-8") as f:
        json.dump({
            "mission": "Chandrayaan-2",
            "instrument": "TMC-2",
            "sensor": "Terrain Mapping Camera-2",
            "gsd": 5.0,
            "sun_azimuth": 45.0,
            "sun_elevation": 30.0,
            "incidence_angle": 60.0,
            "emission_angle": 0.0,
            "phase_angle": 60.0,
            "wavelength": "Panchromatic (500-850 nm)",
            "description": "Simulated Chandrayaan-2 TMC-2 regional context frame",
        }, f, indent=2)

    # 2. Extreme Sun-Angle Inversion Pair (180 deg azimuth delta)
    img_sun_a = render_photometric_shading(dem_cratered, sun_azimuth_deg=45.0, sun_elevation_deg=25.0)
    img_sun_b = render_photometric_shading(dem_cratered, sun_azimuth_deg=225.0, sun_elevation_deg=25.0)

    tifffile.imwrite(SAMPLES_DIR / "sample_sun_azimuth_45deg.tif", img_sun_a)
    with open(SAMPLES_DIR / "sample_sun_azimuth_45deg.json", "w", encoding="utf-8") as f:
        json.dump({
            "mission": "LRO",
            "instrument": "LROC NAC",
            "sensor": "Narrow Angle Camera",
            "gsd": 0.5,
            "sun_azimuth": 45.0,
            "sun_elevation": 25.0,
            "incidence_angle": 65.0,
            "phase_angle": 65.0,
            "description": "Morning sun pass with illumination from North-East (45 deg)",
        }, f, indent=2)

    tifffile.imwrite(SAMPLES_DIR / "sample_sun_azimuth_225deg.tif", img_sun_b)
    with open(SAMPLES_DIR / "sample_sun_azimuth_225deg.json", "w", encoding="utf-8") as f:
        json.dump({
            "mission": "SELENE",
            "instrument": "Kaguya TC",
            "sensor": "Terrain Camera",
            "gsd": 10.0,
            "sun_azimuth": 225.0,
            "sun_elevation": 25.0,
            "incidence_angle": 65.0,
            "phase_angle": 65.0,
            "description": "Afternoon sun pass with illumination from South-West (225 deg - 180 deg inverted shadows)",
        }, f, indent=2)

    # 3. Low-Texture Mare Terrain Pair
    dem_mare = generate_lunar_terrain(size=size, seed=seed + 99, crater_density="sparse")
    mare_base = render_photometric_shading(dem_mare, sun_azimuth_deg=60.0, sun_elevation_deg=40.0)

    M_mare = cv2.getRotationMatrix2D(center, 15.0, 1.1)
    M_mare[0, 2] += 8.0
    mare_rot = cv2.warpAffine(mare_base, M_mare, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    tifffile.imwrite(SAMPLES_DIR / "sample_mare_source.tif", mare_rot)
    tifffile.imwrite(SAMPLES_DIR / "sample_mare_reference.tif", mare_base)

    print("Sample generation complete. Files created:")
    for p in SAMPLES_DIR.glob("sample_*.*"):
        print(f" - {p.name} ({p.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_samples()
