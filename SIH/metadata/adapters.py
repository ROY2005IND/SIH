"""Instrument and mission metadata adapters for Chandrayaan-2 and reference datasets."""
from abc import ABC, abstractmethod
from typing import Any

class BaseMetadataAdapter(ABC):
    """Abstract base metadata adapter ensuring standard representation."""

    @abstractmethod
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        """Convert raw metadata dictionary to standardized LunarImage metadata."""
        pass

class OHRCAdapter(BaseMetadataAdapter):
    """
    Chandrayaan-2 Orbiter High Resolution Camera (OHRC).
    Panchromatic high-resolution imager (~0.25 - 0.32 m GSD at 100 km orbit).
    """
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": "Chandrayaan-2",
            "instrument": "OHRC",
            "sensor": "Orbiter High Resolution Camera",
            "gsd": raw_meta.get("gsd", 0.25),
            "projection": raw_meta.get("projection", "Lunar Simple Cylindrical / Equirectangular"),
            "acquisition_time": raw_meta.get("acquisition_time", "Not available"),
            "sun_azimuth": raw_meta.get("sun_azimuth", "Not available"),
            "sun_elevation": raw_meta.get("sun_elevation", "Not available"),
            "incidence_angle": raw_meta.get("incidence_angle", "Not available"),
            "emission_angle": raw_meta.get("emission_angle", "Not available"),
            "phase_angle": raw_meta.get("phase_angle", "Not available"),
            "wavelength": raw_meta.get("wavelength", "450 - 900 nm (Panchromatic)"),
            "nodata_value": raw_meta.get("nodata_value", 0),
        }

class TMC2Adapter(BaseMetadataAdapter):
    """
    Chandrayaan-2 Terrain Mapping Camera 2 (TMC-2).
    Stereo triplet imager (~5.0 m GSD, Fore/After/Nadir views).
    """
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": "Chandrayaan-2",
            "instrument": "TMC-2",
            "sensor": raw_meta.get("sensor", "Terrain Mapping Camera 2 (Nadir)"),
            "gsd": raw_meta.get("gsd", 5.0),
            "projection": raw_meta.get("projection", "Lunar Simple Cylindrical / Polar Stereographic"),
            "acquisition_time": raw_meta.get("acquisition_time", "Not available"),
            "sun_azimuth": raw_meta.get("sun_azimuth", "Not available"),
            "sun_elevation": raw_meta.get("sun_elevation", "Not available"),
            "incidence_angle": raw_meta.get("incidence_angle", "Not available"),
            "emission_angle": raw_meta.get("emission_angle", "Not available"),
            "phase_angle": raw_meta.get("phase_angle", "Not available"),
            "wavelength": raw_meta.get("wavelength", "500 - 850 nm (Panchromatic)"),
            "nodata_value": raw_meta.get("nodata_value", 0),
        }

class IIRSAdapter(BaseMetadataAdapter):
    """
    Chandrayaan-2 Imaging Infra-Red Spectrometer (IIRS).
    Hyperspectral instrument (~80 m GSD, 250 contiguous bands from 800 - 5000 nm).
    """
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": "Chandrayaan-2",
            "instrument": "IIRS",
            "sensor": "Imaging Infra-Red Spectrometer",
            "gsd": raw_meta.get("gsd", 80.0),
            "projection": raw_meta.get("projection", "Lunar Simple Cylindrical"),
            "acquisition_time": raw_meta.get("acquisition_time", "Not available"),
            "sun_azimuth": raw_meta.get("sun_azimuth", "Not available"),
            "sun_elevation": raw_meta.get("sun_elevation", "Not available"),
            "incidence_angle": raw_meta.get("incidence_angle", "Not available"),
            "emission_angle": raw_meta.get("emission_angle", "Not available"),
            "phase_angle": raw_meta.get("phase_angle", "Not available"),
            "wavelength": raw_meta.get("wavelength", "800 - 5000 nm (250 spectral bands)"),
            "nodata_value": raw_meta.get("nodata_value", -9999),
        }

class LRONACAdapter(BaseMetadataAdapter):
    """
    Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC).
    Panchromatic high-resolution reference (~0.5 - 1.5 m GSD).
    """
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": "Lunar Reconnaissance Orbiter (LRO)",
            "instrument": "LROC NAC",
            "sensor": raw_meta.get("sensor", "Narrow Angle Camera (NAC-L/R)"),
            "gsd": raw_meta.get("gsd", 0.5),
            "projection": raw_meta.get("projection", "Lunar Equirectangular / Polar Stereographic"),
            "acquisition_time": raw_meta.get("acquisition_time", "Not available"),
            "sun_azimuth": raw_meta.get("sun_azimuth", "Not available"),
            "sun_elevation": raw_meta.get("sun_elevation", "Not available"),
            "incidence_angle": raw_meta.get("incidence_angle", "Not available"),
            "emission_angle": raw_meta.get("emission_angle", "Not available"),
            "phase_angle": raw_meta.get("phase_angle", "Not available"),
            "wavelength": raw_meta.get("wavelength", "400 - 750 nm (Visible)"),
            "nodata_value": raw_meta.get("nodata_value", 0),
        }

class SELENETCAdapter(BaseMetadataAdapter):
    """
    SELENE / Kaguya Terrain Camera (TC).
    Panchromatic stereo camera (~10 m GSD).
    """
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": "SELENE / Kaguya",
            "instrument": "Terrain Camera (TC)",
            "sensor": "TC Morning / Evening",
            "gsd": raw_meta.get("gsd", 10.0),
            "projection": raw_meta.get("projection", "Lunar Simple Cylindrical"),
            "acquisition_time": raw_meta.get("acquisition_time", "Not available"),
            "sun_azimuth": raw_meta.get("sun_azimuth", "Not available"),
            "sun_elevation": raw_meta.get("sun_elevation", "Not available"),
            "incidence_angle": raw_meta.get("incidence_angle", "Not available"),
            "emission_angle": raw_meta.get("emission_angle", "Not available"),
            "phase_angle": raw_meta.get("phase_angle", "Not available"),
            "wavelength": raw_meta.get("wavelength", "430 - 850 nm"),
            "nodata_value": raw_meta.get("nodata_value", 0),
        }

class SyntheticAdapter(BaseMetadataAdapter):
    """Controlled Synthetic Lunar Benchmark adapter with known ground-truth parameters."""
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": "Synthetic Benchmark",
            "instrument": raw_meta.get("instrument", "Controlled Procedural Lunar Engine"),
            "sensor": raw_meta.get("sensor", "Simulated Photometric Sensor"),
            "gsd": raw_meta.get("gsd", 1.0),
            "projection": "Planar Orthographic (Synthetic GT)",
            "acquisition_time": "Controlled Simulation Run",
            "sun_azimuth": raw_meta.get("sun_azimuth", 45.0),
            "sun_elevation": raw_meta.get("sun_elevation", 30.0),
            "incidence_angle": raw_meta.get("incidence_angle", 60.0),
            "emission_angle": raw_meta.get("emission_angle", 0.0),
            "phase_angle": raw_meta.get("phase_angle", 60.0),
            "wavelength": "Simulated Visible Regolith (550 nm)",
            "nodata_value": None,
        }

class GenericAdapter(BaseMetadataAdapter):
    """Fallback adapter for generic user images without known mission headers."""
    def adapt(self, raw_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": raw_meta.get("mission", "Not available"),
            "instrument": raw_meta.get("instrument", "Not available"),
            "sensor": raw_meta.get("sensor", "Not available"),
            "gsd": raw_meta.get("gsd", "Not available"),
            "projection": raw_meta.get("projection", "Not available"),
            "acquisition_time": raw_meta.get("acquisition_time", "Not available"),
            "sun_azimuth": raw_meta.get("sun_azimuth", "Not available"),
            "sun_elevation": raw_meta.get("sun_elevation", "Not available"),
            "incidence_angle": raw_meta.get("incidence_angle", "Not available"),
            "emission_angle": raw_meta.get("emission_angle", "Not available"),
            "phase_angle": raw_meta.get("phase_angle", "Not available"),
            "wavelength": raw_meta.get("wavelength", "Not available"),
            "nodata_value": raw_meta.get("nodata_value", None),
        }

def detect_sensor_adapter(hint_str: str) -> BaseMetadataAdapter:
    """Detect appropriate adapter from filename or metadata header hint."""
    hint = hint_str.upper()
    if "OHRC" in hint:
        return OHRCAdapter()
    if "TMC" in hint:
        return TMC2Adapter()
    if "IIRS" in hint:
        return IIRSAdapter()
    if "NAC" in hint or "LRO" in hint:
        return LRONACAdapter()
    if "SELENE" in hint or "KAGUYA" in hint or "TC" in hint:
        return SELENETCAdapter()
    if "SYNTHETIC" in hint or "DEMO" in hint:
        return SyntheticAdapter()
    return GenericAdapter()
