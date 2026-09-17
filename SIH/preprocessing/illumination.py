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

PreprocessingMode = Literal["original", "clahe", "gradient", "edge", "phase", "destripe"]

def apply_destriping(image_u8: np.ndarray) -> np.ndarray:
    """
    Push-broom sensor destriping & vertical line artifact removal.
    Eliminates detector column gain noise in raw uncalibrated orbital rasters.
    """
    col_means = np.mean(image_u8.astype(np.float32), axis=0, keepdims=True)
    global_mean = np.mean(col_means)

    # Subtract column-wise striping bias
    destriped = image_u8.astype(np.float32) - col_means + global_mean
    destriped = np.clip(destriped, 0, 255).astype(np.uint8)

    # Apply CLAHE to accentuate true crater relief after destriping
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(destriped)

def detect_vertical_striping(image_u8: np.ndarray) -> bool:
    """Detect if image contains severe vertical push-broom sensor striping noise."""
    col_std = float(np.std(np.mean(image_u8.astype(np.float32), axis=0)))
    row_std = float(np.std(np.mean(image_u8.astype(np.float32), axis=1)))
    return col_std > 2.0 * (row_std + 1e-5)

def apply_clahe(
    image_u8: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization to accentuate localized crater relief."""
    if detect_vertical_striping(image_u8):
        image_u8 = apply_destriping(image_u8)
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
    if detect_vertical_striping(image_u8):
        image_u8 = apply_destriping(image_u8)
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
    if detect_vertical_striping(image_u8):
        image_u8 = apply_destriping(image_u8)
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
    if detect_vertical_striping(image_u8):
        image_u8 = apply_destriping(image_u8)
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

    if mode == "destripe":
        return apply_destriping(image_u8)
    elif mode == "clahe":
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
        if detect_vertical_striping(image_u8):
            return apply_destriping(image_u8)
        return image_u8.copy()
    else:
        return apply_clahe(image_u8)
        # Fallback to original
        return image_u8.copy()
