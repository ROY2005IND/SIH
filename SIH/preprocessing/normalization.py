"""
Radiometric Normalization and Contrast Stretching Module for Lunar Imagery.
Supports multi-sensor dynamic range standardization and histogram matching.
"""
import numpy as np
import cv2

def percentile_stretch(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robust percentile contrast stretch mapping [p_low, p_high] to [0, 255].
    Suppresses anomalous cosmic ray spikes and shadow saturation.
    """
    img_float = image.astype(np.float32)
    valid_mask = ~np.isnan(img_float)
    if not np.any(valid_mask):
        return np.zeros_like(image, dtype=np.uint8)

    p_low, p_high = np.percentile(img_float[valid_mask], (low_percentile, high_percentile))
    if p_high > p_low:
        stretched = (img_float - p_low) / (p_high - p_low) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)

def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Standard linear min-max stretch to [0, 1] float32."""
    img_float = image.astype(np.float32)
    min_v, max_v = float(np.nanmin(img_float)), float(np.nanmax(img_float))
    if max_v > min_v:
        return (img_float - min_v) / (max_v - min_v)
    return np.zeros_like(img_float)

def match_radiometric_histogram(source_u8: np.ndarray, reference_u8: np.ndarray) -> np.ndarray:
    """
    Align source cumulative distribution function (CDF) to reference CDF.
    Reduces global albedo and sensor sensitivity mismatch prior to correspondence matching.
    """
    src_flat = source_u8.ravel()
    ref_flat = reference_u8.ravel()

    # Compute empirical CDFs
    s_values, bin_idx, s_counts = np.unique(src_flat, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref_flat, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / src_flat.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / ref_flat.size

    # Map quantiles
    interp_values = np.interp(s_quantiles, r_quantiles, r_values)
    matched = interp_values[bin_idx].reshape(source_u8.shape)
    return np.clip(matched, 0, 255).astype(np.uint8)
