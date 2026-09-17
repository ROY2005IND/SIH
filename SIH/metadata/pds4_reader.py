"""
PDS4 XML Label Reader for Chandrayaan-2 and Planetary Data System Observational Products.
Extracts observational geometry, instrument telemetry, and raster dimensions without fabrication.

Status: EXPERIMENTAL — partial parser for PDS4 XML labels.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import xml.etree.ElementTree as ET

try:
    import defusedxml.ElementTree as dET
except ImportError:
    dET = ET

from utils.logging import get_logger

logger = get_logger("PDS4Reader")


@dataclass
class PDS4Metadata:
    """Extracted PDS4 metadata container with provenance tracking."""
    logical_identifier: str | None = None
    product_id: str | None = None
    version_id: str | None = None
    title: str | None = None
    mission: str | None = None
    instrument: str | None = None
    sensor: str | None = None
    acquisition_time: str | None = None
    stop_time: str | None = None
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    data_type: str | None = None
    gsd: float | None = None
    sun_azimuth: float | None = None
    sun_elevation: float | None = None
    incidence_angle: float | None = None
    emission_angle: float | None = None
    phase_angle: float | None = None
    projection: str | None = None
    provenance: str = "PDS4_XML"
    raw_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for merging into LunarImage metadata."""
        out: dict[str, Any] = {
            "pds4_logical_identifier": self.logical_identifier,
            "pds4_product_id": self.product_id,
            "pds4_version_id": self.version_id,
            "pds4_title": self.title,
            "gsd_provenance": "PDS4_XML" if self.gsd is not None else "UNKNOWN",
        }
        if self.mission is not None:
            out["mission"] = self.mission
        if self.instrument is not None:
            out["instrument"] = self.instrument
        if self.sensor is not None:
            out["sensor"] = self.sensor
        if self.acquisition_time is not None:
            out["acquisition_time"] = self.acquisition_time
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        if self.bands is not None:
            out["bands"] = self.bands
        if self.data_type is not None:
            out["data_type"] = self.data_type
        if self.gsd is not None:
            out["gsd"] = self.gsd
        if self.sun_azimuth is not None:
            out["sun_azimuth"] = self.sun_azimuth
        if self.sun_elevation is not None:
            out["sun_elevation"] = self.sun_elevation
        if self.incidence_angle is not None:
            out["incidence_angle"] = self.incidence_angle
        if self.emission_angle is not None:
            out["emission_angle"] = self.emission_angle
        if self.phase_angle is not None:
            out["phase_angle"] = self.phase_angle
        if self.projection is not None:
            out["projection"] = self.projection
        return out


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix if present."""
    return tag.split("}")[-1] if "}" in tag else tag


def _find_text(root: ET.Element, target_local_names: list[str]) -> str | None:
    """Search for first element matching any of the local names and return stripped text."""
    target_set = {n.lower() for n in target_local_names}
    for elem in root.iter():
        if _strip_ns(elem.tag).lower() in target_set and elem.text:
            val = elem.text.strip()
            if val:
                return val
    return None


def _find_float(root: ET.Element, target_local_names: list[str]) -> float | None:
    """Search for first element matching target names and parse as float."""
    txt = _find_text(root, target_local_names)
    if txt is not None:
        try:
            return float(txt)
        except ValueError:
            pass
    return None


def _find_int(root: ET.Element, target_local_names: list[str]) -> int | None:
    """Search for first element matching target names and parse as int."""
    txt = _find_text(root, target_local_names)
    if txt is not None:
        try:
            return int(txt)
        except ValueError:
            pass
    return None


def parse_pds4_label(label_path: Path | str) -> PDS4Metadata | None:
    """
    Parse a PDS4 XML label file safely and extract standardized metadata.

    Returns:
        PDS4Metadata object if parsed successfully, or None if file cannot be parsed.
    """
    path = Path(label_path)
    if not path.exists():
        return None

    try:
        tree = dET.parse(str(path))
        root = tree.getroot()
    except Exception as e:
        logger.warning(f"Failed to parse PDS4 label XML at {path}: {e}")
        return None

    meta = PDS4Metadata()

    # Identification Area
    meta.logical_identifier = _find_text(root, ["logical_identifier"])
    if meta.logical_identifier:
        meta.product_id = meta.logical_identifier.split(":")[-1]
    else:
        meta.product_id = _find_text(root, ["product_id"])

    meta.version_id = _find_text(root, ["version_id"])
    meta.title = _find_text(root, ["title"])

    # Investigation / Mission
    meta.mission = _find_text(root, ["investigation_name", "mission_name"])
    if not meta.mission:
        # Check Observing_System or Investigation_Area names
        for elem in root.iter():
            local = _strip_ns(elem.tag).lower()
            if local in ("investigation_area", "context_area"):
                name = _find_text(elem, ["name"])
                if name:
                    meta.mission = name
                    break

    # Instrument / Sensor
    meta.instrument = _find_text(root, ["instrument_name", "observing_system_component_name"])
    if not meta.instrument:
        for elem in root.iter():
            local = _strip_ns(elem.tag).lower()
            if local in ("observing_system_component",):
                name = _find_text(elem, ["name"])
                comp_type = _find_text(elem, ["type"])
                if comp_type and "instrument" in comp_type.lower() and name:
                    meta.instrument = name
                    break
                elif name and not meta.instrument:
                    meta.instrument = name

    # Time coordinates
    meta.acquisition_time = _find_text(root, ["start_date_time", "observation_time", "start_time"])
    meta.stop_time = _find_text(root, ["stop_date_time", "stop_time"])

    # Geometry & Solar illumination
    meta.sun_azimuth = _find_float(root, ["solar_azimuth_angle", "sun_azimuth", "sub_solar_azimuth"])
    meta.sun_elevation = _find_float(root, ["solar_elevation_angle", "sun_elevation"])
    meta.incidence_angle = _find_float(root, ["incidence_angle", "solar_incidence_angle"])
    meta.emission_angle = _find_float(root, ["emission_angle"])
    meta.phase_angle = _find_float(root, ["phase_angle"])

    # GSD / Pixel Resolution
    meta.gsd = _find_float(
        root,
        [
            "pixel_resolution",
            "ground_sample_distance",
            "horizontal_pixel_resolution",
            "pixel_scale",
            "map_scale",
        ],
    )

    # Array 2D / 3D dimensions
    # Search for Axis_Array elements with sequence_number or axis_name
    axes_elements: list[int] = []
    for elem in root.iter():
        if _strip_ns(elem.tag).lower() == "axis_array":
            n_elems = _find_int(elem, ["elements"])
            if n_elems is not None:
                axes_elements.append(n_elems)

    if len(axes_elements) >= 2:
        meta.height = axes_elements[0]
        meta.width = axes_elements[1]
        if len(axes_elements) >= 3:
            meta.bands = axes_elements[2]
    else:
        meta.width = _find_int(root, ["width", "samples", "line_samples"])
        meta.height = _find_int(root, ["height", "lines"])

    meta.data_type = _find_text(root, ["data_type", "data_type_name"])
    meta.projection = _find_text(root, ["projection_name", "map_projection_name", "coordinate_system_name"])

    return meta
