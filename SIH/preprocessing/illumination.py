"""
Illumination-Invariant Image Preprocessing Engine.
Provides representations designed to withstand extreme solar elevation and azimuth shadow shifts:
- CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- Sobel Gradient Magnitude
- Edge (Morphological / Canny)
- Phase-based Local Energy representation
"""
from typing import Literal
import numpy as np
import cv2

PreprocessingMode = Literal["original", "clahe", "gradient", "edge", "phase"]

def apply_clahe(
    image_u8: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization to accentuate localized crater relief."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_u8)

def apply_gradient_magnitude(
    image_u8: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Compute Sobel gradient magnitude: G = sqrt(Gx^2 + Gy^2).
    Inverts shadows and provides structural invariance against changing solar incidence angles.
    """
    gx = cv2.Sobel(image_u8, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image_u8, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def apply_edge_representation(
    image_u8: np.ndarray,
    low_thresh: int = 40,
    high_thresh: int = 120,
) -> np.ndarray:
    """Canny edge representation extracting discrete topographic ridge and crater rim boundaries."""
    return cv2.Canny(image_u8, low_thresh, high_thresh)

def apply_phase_congruency(
    image_u8: np.ndarray,
    n_scales: int = 3,
) -> np.ndarray:
    """
    Phase-based local energy representation via 2D Riesz transform / quadrature filter approximation.
    NOTE: Documented limitation: Phase-based representations are invariant to contrast polarity
    and illumination changes, but can be sensitive to high-frequency speckle noise.
    """
    f = np.fft.fft2(image_u8.astype(np.float32))
    fshift = np.fft.fftshift(f)
    rows, cols = image_u8.shape
    crow, ccol = rows // 2, cols // 2

    # Bandpass filter around dominant lunar topographic spatial frequencies
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2) + 1e-6

    # Butterworth bandpass filter
    r_low = max(rows, cols) * 0.02
    r_high = max(rows, cols) * 0.35
    bandpass = 1.0 / (1.0 + (r_low / r) ** 4) * (1.0 / (1.0 + (r / r_high) ** 4))

    filtered_shift = fshift * bandpass
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    energy = np.abs(filtered)

    norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def apply_preprocessing(
    image_u8: np.ndarray,
    method: PreprocessingMode = "clahe",
    params: dict | None = None,
) -> np.ndarray:
    """Master preprocessing dispatcher."""
    p = params or {}
    mode = method.lower()

    if mode == "clahe":
        clip = p.get("clip_limit", 2.5)
        grid = tuple(p.get("tile_grid_size", (8, 8)))
        return apply_clahe(image_u8, clip_limit=clip, tile_grid_size=grid)
    elif mode == "gradient":
        ksize = p.get("sobel_ksize", 3)
        return apply_gradient_magnitude(image_u8, ksize=ksize)
    elif mode == "edge":
        low = p.get("canny_low", 40)
        high = p.get("canny_high", 120)
        return apply_edge_representation(image_u8, low_thresh=low, high_thresh=high)
    elif mode == "phase":
        return apply_phase_congruency(image_u8)
    elif mode == "original":
        return image_u8.copy()
    else:
        # Fallback to original
        return image_u8.copy()
