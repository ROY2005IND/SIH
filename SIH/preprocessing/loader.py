"""Robust multi-format image loader and standardized LunarImage data structure."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import cv2
import tifffile
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

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
    gsd_provenance: str = "UNKNOWN"
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
        Memory-optimized for gigapixel rasters.
        """
        img = self.image

        # Fast-path for single-channel uint8 image with no nodata mask
        if img.dtype == np.uint8 and img.ndim == 2 and self.nodata_value is None:
            return img

        # Extract 2D array if 3D
        if img.ndim == 3:
            if img.shape[2] == 1:
                img_2d = img[:, :, 0]
            elif img.shape[2] == 3:
                img_2d = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                mid_band = img.shape[2] // 2
                img_2d = img[:, :, mid_band]
        else:
            img_2d = img

        if img_2d.dtype == np.uint8 and self.nodata_value is None:
            return img_2d

        # Determine valid mask using subsampling for large images
        h, w = img_2d.shape[:2]
        total_pixels = h * w

        if self.nodata_value is not None:
            nodata_val = self.nodata_value
            if np.isnan(nodata_val):
                valid_mask = ~np.isnan(img_2d)
            else:
                valid_mask = (img_2d != nodata_val) & (~np.isnan(img_2d))
        else:
            valid_mask = ~np.isnan(img_2d) if np.issubdtype(img_2d.dtype, np.floating) else None

        # For large images (>2M px), compute percentiles on a subsampled view to save memory
        if total_pixels > 2_000_000:
            step = max(1, int(np.sqrt(total_pixels / 1_000_000)))
            sub_img = img_2d[::step, ::step]
            sub_mask = valid_mask[::step, ::step] if valid_mask is not None else None
            valid_sample = sub_img[sub_mask] if sub_mask is not None else sub_img.ravel()
        else:
            valid_sample = img_2d[valid_mask] if valid_mask is not None else img_2d.ravel()

        if len(valid_sample) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        if robust_percentile and len(valid_sample) > 10:
            p_low, p_high = float(np.percentile(valid_sample, 1.0)), float(np.percentile(valid_sample, 99.0))
        else:
            p_low, p_high = float(np.min(valid_sample)), float(np.max(valid_sample))

        if p_high <= p_low:
            return np.zeros((h, w), dtype=np.uint8)

        # Efficient uint8 conversion without double float64 allocation
        scale = 255.0 / (p_high - p_low)
        img_float = img_2d.astype(np.float32)
        np.clip(img_float, p_low, p_high, out=img_float)
        img_float -= p_low
        img_float *= scale
        return img_float.astype(np.uint8)

    def metadata_dict(self) -> dict[str, Any]:
        """Export standardized metadata dictionary."""
        return {
            "mission": self.mission,
            "instrument": self.instrument,
            "sensor": self.sensor,
            "gsd_m_px": self.gsd,
            "gsd_provenance": self.gsd_provenance,
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
            gsd_provenance=meta.get("gsd_provenance", "SYNTHETIC_GT" if "gsd" in meta else "UNKNOWN"),
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
    arr = None
    if suffix in [".tif", ".tiff", ".geotiff"]:
        try:
            arr = tifffile.imread(path)
        except Exception:
            # Fallback to OpenCV / PIL if tifffile lacks specific compression codecs
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is not None and arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif arr is None:
                try:
                    from PIL import Image
                    with Image.open(path) as img:
                        arr = np.array(img)
                except Exception as e:
                    raise ValueError(f"Failed to decode compressed TIFF image from {path}: {e}")

    if arr is None:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            try:
                from PIL import Image
                with Image.open(path) as img:
                    arr = np.array(img)
            except Exception:
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
        gsd_provenance=parsed_meta.get("gsd_provenance", "UNKNOWN"),
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
