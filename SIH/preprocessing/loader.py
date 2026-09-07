"""Robust multi-format image loader and standardized LunarImage data structure."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import cv2
import tifffile
from metadata.reader import parse_lunar_metadata

@dataclass
class LunarImage:
    """Standardized planetary lunar raster object encapsulating data and metadata."""
    image: np.ndarray
    width: int
    height: int
    channels: int
    mission: str = "Not available"
    instrument: str = "Not available"
    sensor: str = "Not available"
    gsd: float | str = "Not available"
    projection: str = "Not available"
    acquisition_time: str = "Not available"
    sun_azimuth: float | str = "Not available"
    sun_elevation: float | str = "Not available"
    incidence_angle: float | str = "Not available"
    emission_angle: float | str = "Not available"
    phase_angle: float | str = "Not available"
    wavelength: str = "Not available"
    nodata_value: float | None = None
    source_path: str = "In-Memory / Synthetic"
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def to_uint8(self, robust_percentile: bool = True) -> np.ndarray:
        """
        Convert image to 8-bit uint8 grayscale [0, 255] for feature extractors.
        Uses 1st to 99th percentile contrast stretch to handle high-dynamic-range lunar sensors.
        """
        img = self.image
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # Hyperspectral cube - default to center band
                mid_band = img.shape[2] // 2
                img = img[:, :, mid_band]

        img_float = img.astype(np.float32)

        # Mask nodata if present
        if self.nodata_value is not None:
            valid_mask = (img != self.nodata_value) & (~np.isnan(img))
        else:
            valid_mask = ~np.isnan(img)

        if not np.any(valid_mask):
            return np.zeros((self.height, self.width), dtype=np.uint8)

        valid_pixels = img_float[valid_mask]

        if robust_percentile and len(valid_pixels) > 10:
            p_low, p_high = np.percentile(valid_pixels, (1.0, 99.0))
            if p_high > p_low:
                stretched = np.clip((img_float - p_low) / (p_high - p_low) * 255.0, 0, 255)
            else:
                stretched = np.zeros_like(img_float)
        else:
            min_val, max_val = float(np.min(valid_pixels)), float(np.max(valid_pixels))
            if max_val > min_val:
                stretched = ((img_float - min_val) / (max_val - min_val) * 255.0).clip(0, 255)
            else:
                stretched = np.zeros_like(img_float)

        return stretched.astype(np.uint8)

    def metadata_dict(self) -> dict[str, Any]:
        """Export standardized metadata dictionary."""
        return {
            "mission": self.mission,
            "instrument": self.instrument,
            "sensor": self.sensor,
            "gsd_m_px": self.gsd,
            "projection": self.projection,
            "acquisition_time": self.acquisition_time,
            "sun_azimuth_deg": self.sun_azimuth,
            "sun_elevation_deg": self.sun_elevation,
            "incidence_angle_deg": self.incidence_angle,
            "emission_angle_deg": self.emission_angle,
            "phase_angle_deg": self.phase_angle,
            "wavelength": self.wavelength,
            "dimensions": f"{self.width} x {self.height}",
            "channels": self.channels,
            "source_path": self.source_path,
        }

def load_lunar_image(
    source: Path | str | np.ndarray,
    metadata_override: dict[str, Any] | None = None,
) -> LunarImage:
    """
    Load an image from file (GeoTIFF, TIFF, PNG, JPEG) or NumPy array into a standardized LunarImage.
    Preserves exact radiometric depth and extracts all metadata without destructive downsampling.
    """
    if isinstance(source, np.ndarray):
        arr = source
        h, w = arr.shape[:2]
        c = 1 if arr.ndim == 2 else arr.shape[2]
        meta = metadata_override or {}
        return LunarImage(
            image=arr,
            width=w,
            height=h,
            channels=c,
            mission=meta.get("mission", "Synthetic"),
            instrument=meta.get("instrument", "Procedural Generator"),
            sensor=meta.get("sensor", "Simulated Sensor"),
            gsd=meta.get("gsd", 1.0),
            projection=meta.get("projection", "Planar Orthographic"),
            acquisition_time=meta.get("acquisition_time", "Simulated"),
            sun_azimuth=meta.get("sun_azimuth", "Not available"),
            sun_elevation=meta.get("sun_elevation", "Not available"),
            incidence_angle=meta.get("incidence_angle", "Not available"),
            emission_angle=meta.get("emission_angle", "Not available"),
            phase_angle=meta.get("phase_angle", "Not available"),
            wavelength=meta.get("wavelength", "Not available"),
            nodata_value=meta.get("nodata_value", None),
            source_path="In-Memory Array",
            auxiliary=meta,
        )

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Lunar raster file not found: {path}")

    # Read image data
    suffix = path.suffix.lower()
    if suffix in [".tif", ".tiff", ".geotiff"]:
        arr = tifffile.imread(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Failed to decode image from {path}")
        if arr.ndim == 3 and arr.shape[2] == 3:
            # OpenCV loads BGR, convert to RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    h, w = arr.shape[:2]
    c = 1 if arr.ndim == 2 else arr.shape[2]

    # Parse metadata
    parsed_meta = parse_lunar_metadata(path, user_override=metadata_override)

    return LunarImage(
        image=arr,
        width=w,
        height=h,
        channels=c,
        mission=parsed_meta.get("mission", "Not available"),
        instrument=parsed_meta.get("instrument", "Not available"),
        sensor=parsed_meta.get("sensor", "Not available"),
        gsd=parsed_meta.get("gsd", "Not available"),
        projection=parsed_meta.get("projection", "Not available"),
        acquisition_time=parsed_meta.get("acquisition_time", "Not available"),
        sun_azimuth=parsed_meta.get("sun_azimuth", "Not available"),
        sun_elevation=parsed_meta.get("sun_elevation", "Not available"),
        incidence_angle=parsed_meta.get("incidence_angle", "Not available"),
        emission_angle=parsed_meta.get("emission_angle", "Not available"),
        phase_angle=parsed_meta.get("phase_angle", "Not available"),
        wavelength=parsed_meta.get("wavelength", "Not available"),
        nodata_value=parsed_meta.get("nodata_value", None),
        source_path=str(path),
        auxiliary=parsed_meta,
    )
