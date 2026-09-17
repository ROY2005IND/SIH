"""Metadata reader and extractor for planetary lunar rasters."""
from pathlib import Path
from typing import Any
import json
import tifffile
from .adapters import detect_sensor_adapter, GenericAdapter
from .pds4_reader import parse_pds4_label


def parse_lunar_metadata(file_path: Path | str, user_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Extract and adapt metadata from a raster file and any associated sidecars.
    Follows rigorous provenance hierarchy:
        USER_OVERRIDE > PDS4_XML > GEOTIFF_TAG > SIDECAR_JSON > UNKNOWN
    Never invents missing values.
    """
    path = Path(file_path)
    raw_meta: dict[str, Any] = {}
    gsd_provenance = "UNKNOWN"
    detection_confidence = "HINT"

    # 1. Check for sidecar JSON metadata (common in intermediate processing)
    sidecar_json = path.with_suffix(".json")
    if sidecar_json.exists():
        try:
            with open(sidecar_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                raw_meta.update(json_data)
                if "gsd" in json_data and json_data["gsd"] is not None:
                    gsd_provenance = "SIDECAR_JSON"
        except Exception:
            pass

    # 2. Inspect TIFF / GeoTIFF tags
    if path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
        try:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                raw_meta["width"] = page.shape[1] if len(page.shape) >= 2 else page.shape[0]
                raw_meta["height"] = page.shape[0]

                # Check GeoTIFF pixel scale tag (Tag 33550: ModelPixelScaleTag)
                if hasattr(page, "geotiff_tags") and page.geotiff_tags:
                    geo_tags = page.geotiff_tags
                    # ModelPixelScaleTag: [scale_x, scale_y, scale_z]
                    if 33550 in geo_tags:
                        raw_meta["gsd"] = float(geo_tags[33550][0])
                        gsd_provenance = "GEOTIFF_TAG"
        except Exception:
            pass

    # 3. Check for PDS4 XML label (.xml or .lbl sidecar, or file itself)
    pds4_meta = None
    xml_candidate = None
    if path.suffix.lower() == ".xml":
        xml_candidate = path
    elif path.with_suffix(".xml").exists():
        xml_candidate = path.with_suffix(".xml")
    elif path.with_suffix(".lbl").exists():
        xml_candidate = path.with_suffix(".lbl")

    if xml_candidate is not None:
        pds4_meta = parse_pds4_label(xml_candidate)
        if pds4_meta is not None:
            pds4_dict = pds4_meta.to_dict()
            raw_meta.update(pds4_dict)
            if pds4_meta.gsd is not None:
                gsd_provenance = "PDS4_XML"
            if pds4_meta.instrument:
                detection_confidence = "AUTHORITATIVE"

    # 4. Apply user overrides if provided (highest priority)
    if user_override:
        raw_meta.update(user_override)
        if "gsd" in user_override:
            gsd_provenance = "USER_OVERRIDE"
        if "instrument" in user_override or "sensor" in user_override:
            detection_confidence = "USER_OVERRIDE"

    raw_meta["gsd_provenance"] = gsd_provenance

    # 5. Detect adapter based on filename or sensor hint
    hint = f"{path.name} {raw_meta.get('sensor', '')} {raw_meta.get('instrument', '')} {raw_meta.get('mission', '')}"
    adapter = detect_sensor_adapter(hint)
    standardized = adapter.adapt(raw_meta)
    standardized["source_path"] = str(path)
    standardized["detection_confidence"] = detection_confidence
    standardized["gsd_provenance"] = gsd_provenance

    return standardized
