"""Metadata reader and extractor for planetary lunar rasters."""
from pathlib import Path
from typing import Any
import json
import tifffile
from .adapters import detect_sensor_adapter, GenericAdapter

def parse_lunar_metadata(file_path: Path | str, user_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Extract and adapt metadata from a raster file and any associated sidecars.
    Never invents values: missing parameters remain 'Not available'.
    """
    path = Path(file_path)
    raw_meta: dict[str, Any] = {}

    # Check for sidecar JSON metadata (common in planetary processing pipelines)
    sidecar_json = path.with_suffix(".json")
    if sidecar_json.exists():
        try:
            with open(sidecar_json, "r", encoding="utf-8") as f:
                raw_meta.update(json.load(f))
        except Exception:
            pass

    # Inspect TIFF / GeoTIFF tags
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
        except Exception:
            pass

    # Apply user overrides if provided
    if user_override:
        raw_meta.update(user_override)

    # Detect adapter based on filename or sensor hint
    hint = f"{path.name} {raw_meta.get('sensor', '')} {raw_meta.get('instrument', '')} {raw_meta.get('mission', '')}"
    adapter = detect_sensor_adapter(hint)
    standardized = adapter.adapt(raw_meta)
    standardized["source_path"] = str(path)

    return standardized
