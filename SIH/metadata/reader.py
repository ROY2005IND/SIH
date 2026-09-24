"""Metadata reader and extractor for planetary lunar rasters."""
from pathlib import Path
from typing import Any
import json
import tifffile
from .adapters import detect_sensor_adapter, GenericAdapter
from .pds4_reader import parse_pds4_label


def _find_pds4_label(path: Path) -> Path | None:
    """Locate an authoritative PDS4 label belonging to *path*.

    ISRO archives do not always give the label the same stem as the raster.  A
    product directory commonly contains ``product.xml`` while the raster is
    named in the label's ``file_name`` element.  Looking only for
    ``path.with_suffix('.xml')`` silently drops the geometry in that very
    common layout.
    """
    direct_candidates = (path.with_suffix(".xml"), path.with_suffix(".lbl"))
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    # Keep the search bounded: product folders are expected to be flat and a
    # label must explicitly name this raster before it is accepted.
    for candidate in list(path.parent.glob("*.xml")) + list(path.parent.glob("*.lbl")):
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path.name.lower() in text.lower():
            return candidate
    return None


def parse_lunar_metadata(file_path: Path | str, user_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Extract and adapt metadata from a raster file and any associated sidecars.
    Follows rigorous provenance hierarchy:
        USER_OVERRIDE > PDS4_XML > GEOTIFF_TAG > SIDECAR_JSON > UNKNOWN
    Never invents missing values.
    """
    path = Path(file_path)
    # This optional UI/API hint permits uploaded PDS4 labels whose filenames
    # cannot be preserved alongside the uploaded raster.  Do not expose it as
    # scientific metadata or allow it to affect adapter detection.
    overrides = dict(user_override or {})
    explicit_label = overrides.pop("pds4_label_path", None)
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
    if explicit_label:
        candidate = Path(explicit_label)
        xml_candidate = candidate if candidate.is_file() else None
    elif path.suffix.lower() == ".xml":
        xml_candidate = path
    else:
        xml_candidate = _find_pds4_label(path)

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
    if overrides:
        raw_meta.update(overrides)
        if "gsd" in overrides:
            gsd_provenance = "USER_OVERRIDE"
        if "instrument" in overrides or "sensor" in overrides:
            detection_confidence = "USER_OVERRIDE"

    raw_meta["gsd_provenance"] = gsd_provenance

    # 5. Detect adapter based on filename or sensor hint
    hint = f"{path.name} {raw_meta.get('sensor', '')} {raw_meta.get('instrument', '')} {raw_meta.get('mission', '')}"
    adapter = detect_sensor_adapter(hint)
    standardized = adapter.adapt(raw_meta)
    # Adapters standardize presentation fields, but geospatial fields (bbox,
    # map projection parameters, PDS identifiers) must survive for overlap
    # analysis and audit/export.  Standardized values take precedence only for
    # the fields the adapter intentionally normalizes.
    standardized = {**raw_meta, **standardized}
    standardized["source_path"] = str(path)
    standardized["detection_confidence"] = detection_confidence
    standardized["gsd_provenance"] = gsd_provenance

    return standardized
