"""
Image Warping, Resampling, and Visual Comparison Product Generation.
Produces registered imagery, radiometric difference maps, and multi-mode overlays.
"""
import numpy as np
import cv2

def warp_image_to_reference(
    source_image: np.ndarray,
    transformation: np.ndarray,
    reference_shape: tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> np.ndarray:
    """
    Warp source image into reference coordinate frame using estimated transformation matrix M.
    Reference shape: (height, width).
    """
    ref_h, ref_w = reference_shape[:2]

    if transformation.shape == (2, 3):
        warped = cv2.warpAffine(
            source_image,
            transformation,
            (ref_w, ref_h),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
    elif transformation.shape == (3, 3):
        warped = cv2.warpPerspective(
            source_image,
            transformation,
            (ref_w, ref_h),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
    else:
        raise ValueError(f"Invalid transformation matrix shape: {transformation.shape}")

    return warped

def compute_difference_map(
    registered_source: np.ndarray,
    reference_image: np.ndarray,
    mask_invalid: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Compute pixel-wise absolute difference map: |I_reg - I_ref|.
    Masks out border zero-padding to calculate mean difference strictly over true overlapping area.
    Returns:
        diff_map: (H, W) uint8
        mean_overlap_diff: float
    """
    reg_f = registered_source.astype(np.float32)
    ref_f = reference_image.astype(np.float32)

    diff = np.abs(reg_f - ref_f)

    if mask_invalid:
        valid_mask = (registered_source > 0) & (reference_image > 0)
        if np.any(valid_mask):
            mean_diff = float(np.mean(diff[valid_mask]))
        else:
            mean_diff = float(np.mean(diff))
    else:
        mean_diff = float(np.mean(diff))

    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    return diff_u8, round(mean_diff, 2)

def create_alpha_overlay(
    registered_source: np.ndarray,
    reference_image: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """
    Generate color-coded false-color alignment overlay:
    Reference in Red channel, Registered Source in Cyan (Green + Blue) channel.
    Perfect registration appears as neutral monochrome; misalignment produces color fringes.
    """
    h, w = reference_image.shape[:2]
    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Reference in Red channel
    overlay_rgb[:, :, 0] = reference_image
    # Registered Source in Green and Blue channels (Cyan)
    overlay_rgb[:, :, 1] = registered_source
    overlay_rgb[:, :, 2] = registered_source

    return overlay_rgb

def create_split_wipe(
    registered_source: np.ndarray,
    reference_image: np.ndarray,
    split_fraction: float = 0.50,
) -> np.ndarray:
    """Create split wipe image with a distinct technical divider line."""
    h, w = reference_image.shape[:2]
    split_x = int(w * np.clip(split_fraction, 0.05, 0.95))

    wipe = np.zeros_like(reference_image)
    # Left side: Registered Source
    wipe[:, :split_x] = registered_source[:, :split_x]
    # Right side: Reference
    wipe[:, split_x:] = reference_image[:, split_x:]

    # Draw vertical divider line
    wipe[:, max(0, split_x - 1):min(w, split_x + 1)] = 255
    return wipe
