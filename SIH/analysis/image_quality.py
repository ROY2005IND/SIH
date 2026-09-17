"""
Planetary Image Quality Analysis Module.
Evaluates radiometric contrast, Shannon texture entropy, valid pixel fraction,
and structural sharpness prior to correspondence matching.
"""
from dataclasses import dataclass, field
import numpy as np
import cv2
from preprocessing.loader import LunarImage
from config import load_settings

@dataclass
class ImageQualityReport:
    """Standardized quality assessment for a single planetary lunar raster."""
    valid_pixel_pct: float
    dynamic_range_min: float
    dynamic_range_max: float
    bit_depth: int
    contrast_std: float
    contrast_level: str        # "HIGH", "MODERATE", "LOW"
    texture_entropy: float     # Shannon entropy in bits
    texture_level: str         # "RICH", "MODERATE", "LOW"
    sharpness_laplacian: float # Variance of Laplacian
    readiness_status: str      # "OPTIMAL", "ACCEPTABLE", "WARNING_LOW_TEXTURE", "WARNING_LOW_CONTRAST"
    warnings: list[str] = field(default_factory=list)

def analyze_image_quality(lunar_image: LunarImage, settings_override: dict | None = None) -> ImageQualityReport:
    """
    Analyze radiometric and textural quality of a lunar image.
    Memory-optimized for gigapixel lunar orbital rasters via spatial subsampling.
    """
    settings = settings_override or load_settings()
    analysis_cfg = settings.get("analysis", {}).get("image_quality", {})
    low_contrast_thresh = analysis_cfg.get("low_contrast_std", 12.0)
    low_entropy_thresh = analysis_cfg.get("low_texture_entropy", 3.2)
    max_nodata_frac = analysis_cfg.get("max_nodata_fraction", 0.40)

    raw = lunar_image.image
    h, w = raw.shape[:2]
    total_pixels = h * w

    # Subsample step for memory-safe statistical analysis on large rasters (>2M pixels)
    step = max(1, int(np.sqrt(total_pixels / 2_000_000))) if total_pixels > 2_000_000 else 1
    raw_sub = raw[::step, ::step]

    # 1. Valid pixel calculation
    if lunar_image.nodata_value is not None:
        nodata_val = lunar_image.nodata_value
        if np.isnan(nodata_val):
            valid_mask = ~np.isnan(raw_sub)
        else:
            valid_mask = (raw_sub != nodata_val) & (~np.isnan(raw_sub))
    else:
        valid_mask = ~np.isnan(raw_sub) if np.issubdtype(raw_sub.dtype, np.floating) else np.ones_like(raw_sub, dtype=bool)

    valid_count = int(np.sum(valid_mask))
    sub_total = raw_sub.size
    valid_pct = round((valid_count / max(sub_total, 1)) * 100.0, 2)

    warnings: list[str] = []
    if (1.0 - valid_count / sub_total) > max_nodata_frac:
        warnings.append(f"High nodata/missing pixel area: {100.0 - valid_pct:.1f}%")

    # 2. Dynamic range (computed on float32 subsample to prevent 1GB+ float64 allocation)
    if valid_count > 0:
        valid_data = raw_sub[valid_mask].astype(np.float32)
        d_min = float(np.min(valid_data))
        d_max = float(np.max(valid_data))
        std_val = float(np.std(valid_data))
    else:
        d_min, d_max, std_val = 0.0, 0.0, 0.0
        warnings.append("No valid radiometric pixels found in image.")

    bit_depth = 16 if raw.dtype in [np.uint16, np.int16] else (32 if raw.dtype in [np.float32, np.int32] else 8)

    # 3. Contrast level & Textural Shannon Entropy on uint8 subsample
    u8 = lunar_image.to_uint8()
    u8_sub = u8[::step, ::step]
    contrast_std = float(np.std(u8_sub))

    if contrast_std >= 35.0:
        contrast_level = "HIGH"
    elif contrast_std >= low_contrast_thresh:
        contrast_level = "MODERATE"
    else:
        contrast_level = "LOW"
        warnings.append(f"Low radiometric contrast detected (std={contrast_std:.1f} < {low_contrast_thresh})")

    # 4. Textural Shannon Entropy
    hist, _ = np.histogram(u8_sub, bins=256, range=(0, 256), density=True)
    hist_nonzero = hist[hist > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero))) if len(hist_nonzero) > 0 else 0.0

    if entropy >= 5.5:
        texture_level = "RICH"
    elif entropy >= low_entropy_thresh:
        texture_level = "MODERATE"
    else:
        texture_level = "LOW"
        warnings.append(f"Low textural entropy detected (H={entropy:.2f} bits < {low_entropy_thresh})")

    # 5. Sharpness (Variance of Laplacian on subsampled uint8)
    laplacian = cv2.Laplacian(u8_sub, cv2.CV_32F)
    sharpness = float(laplacian.var())

    # Overall readiness status
    if valid_pct < 50.0:
        readiness = "INSUFFICIENT_VALID_PIXELS"
    elif contrast_level == "LOW" and texture_level == "LOW":
        readiness = "WARNING_LOW_CONTRAST_AND_TEXTURE"
    elif texture_level == "LOW":
        readiness = "WARNING_LOW_TEXTURE"
    elif contrast_level == "LOW":
        readiness = "WARNING_LOW_CONTRAST"
    elif warnings:
        readiness = "ACCEPTABLE"
    else:
        readiness = "OPTIMAL"

    return ImageQualityReport(
        valid_pixel_pct=valid_pct,
        dynamic_range_min=d_min,
        dynamic_range_max=d_max,
        bit_depth=bit_depth,
        contrast_std=round(contrast_std, 2),
        contrast_level=contrast_level,
        texture_entropy=round(entropy, 2),
        texture_level=texture_level,
        sharpness_laplacian=round(sharpness, 2),
        readiness_status=readiness,
        warnings=warnings,
    )
