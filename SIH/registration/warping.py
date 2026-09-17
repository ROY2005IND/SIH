"""
Image Warping, Resampling, and Visual Comparison Product Generation.
Produces registered imagery, radiometric difference maps, and multi-mode overlays.
"""
import numpy as np
import cv2

# Mid-gray fill value used for border regions outside the warped source footprint.
# This makes empty warp canvas areas visually distinguishable from valid (dark) terrain.
_BORDER_FILL = 128


def normalize_for_display(img: np.ndarray, border_fill: int = _BORDER_FILL) -> np.ndarray:
    """
    Auto-stretch an image to full [0, 255] range for display, ignoring border-fill pixels.
    Prevents dark display caused by a narrow effective dynamic range in the valid region.
    """
    if img is None or img.size == 0:
        return img
    img_f = img.astype(np.float32)
    # Build valid pixel mask: exclude the uniform border fill regions
    valid = np.abs(img_f - float(border_fill)) > 2
    if not np.any(valid):
        valid = np.ones_like(img_f, dtype=bool)  # all valid fallback
    p_low  = float(np.percentile(img_f[valid], 1.0))
    p_high = float(np.percentile(img_f[valid], 99.0))
    if p_high <= p_low:
        return img
    stretched = np.clip((img_f - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
    return stretched


def warp_image_to_reference(
    source_image: np.ndarray,
    transformation: np.ndarray,
    reference_shape: tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = _BORDER_FILL,
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


def crop_to_valid_overlap(
    registered_source: np.ndarray,
    reference_image: np.ndarray,
    border_fill: int = _BORDER_FILL,
    margin_px: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop both the registered source and the reference to the bounding box of the
    valid (non-border-fill) content in the registered source.

    When the source image is a small strip warped onto a large reference canvas
    (e.g. OHRC onto a 52224-tall TMC-2 strip), the canvas is mostly empty.
    This function returns only the region that actually contains warped content,
    making the display far more informative.

    Returns (cropped_registered, cropped_reference) — same shape, properly aligned.
    """
    if registered_source is None or registered_source.size == 0:
        return registered_source, reference_image

    h, w = reference_image.shape[:2]
    if registered_source.shape[:2] != (h, w):
        return registered_source, reference_image

    # Valid pixel mask: pixels that differ from the uniform border fill by >4 DN
    valid = np.abs(registered_source.astype(np.int32) - border_fill) > 4

    if not np.any(valid):
        return registered_source, reference_image

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    r_min, r_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c_min, c_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    # Apply generous margin
    r_min = max(0,  r_min - margin_px)
    r_max = min(h - 1, r_max + margin_px)
    c_min = max(0,  c_min - margin_px)
    c_max = min(w - 1, c_max + margin_px)

    return registered_source[r_min:r_max+1, c_min:c_max+1], reference_image[r_min:r_max+1, c_min:c_max+1]


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
    if reference_image.ndim == 3 and reference_image.shape[2] == 3:
        ref_2d = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
    elif reference_image.ndim == 3:
        ref_2d = reference_image[:, :, 0]
    else:
        ref_2d = reference_image

    if registered_source.ndim == 3 and registered_source.shape[2] == 3:
        reg_2d = cv2.cvtColor(registered_source, cv2.COLOR_RGB2GRAY)
    elif registered_source.ndim == 3:
        reg_2d = registered_source[:, :, 0]
    else:
        reg_2d = registered_source

    h, w = ref_2d.shape[:2]
    if reg_2d.shape[:2] != (h, w):
        canvas = np.full((h, w), _BORDER_FILL, dtype=np.uint8)
        rh = min(reg_2d.shape[0], h)
        rw = min(reg_2d.shape[1], w)
        canvas[:rh, :rw] = reg_2d[:rh, :rw]
        reg_2d = canvas

    reg_f = reg_2d.astype(np.float32)
    ref_f = ref_2d.astype(np.float32)

    diff = np.abs(reg_f - ref_f)

    if mask_invalid:
        valid_mask = (reg_2d > 0) & (ref_2d > 0)
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
    if reference_image.ndim == 3 and reference_image.shape[2] == 3:
        ref_2d = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
    elif reference_image.ndim == 3:
        ref_2d = reference_image[:, :, 0]
    else:
        ref_2d = reference_image

    if registered_source.ndim == 3 and registered_source.shape[2] == 3:
        reg_2d = cv2.cvtColor(registered_source, cv2.COLOR_RGB2GRAY)
    elif registered_source.ndim == 3:
        reg_2d = registered_source[:, :, 0]
    else:
        reg_2d = registered_source

    h, w = ref_2d.shape[:2]
    if reg_2d.shape[:2] != (h, w):
        canvas = np.full((h, w), _BORDER_FILL, dtype=np.uint8)
        rh = min(reg_2d.shape[0], h)
        rw = min(reg_2d.shape[1], w)
        canvas[:rh, :rw] = reg_2d[:rh, :rw]
        reg_2d = canvas

    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Reference in Red channel
    overlay_rgb[:, :, 0] = ref_2d
    # Registered Source in Green and Blue channels (Cyan)
    overlay_rgb[:, :, 1] = reg_2d
    overlay_rgb[:, :, 2] = reg_2d

    return overlay_rgb

def create_split_wipe(
    registered_source: np.ndarray,
    reference_image: np.ndarray,
    split_fraction: float = 0.50,
) -> np.ndarray:
    """Create split wipe image with auto-normalized display on both sides and a bright divider."""
    h, w = reference_image.shape[:2]
    split_x = int(w * np.clip(split_fraction, 0.05, 0.95))

    # Auto-stretch both sides independently so neither looks dark
    ref_disp = normalize_for_display(reference_image)
    reg_disp = normalize_for_display(registered_source)

    if ref_disp.ndim == 3 and ref_disp.shape[2] == 3:
        ref_disp = cv2.cvtColor(ref_disp, cv2.COLOR_RGB2GRAY)
    elif ref_disp.ndim == 3:
        ref_disp = ref_disp[:, :, 0]

    if reg_disp.ndim == 3 and reg_disp.shape[2] == 3:
        reg_disp = cv2.cvtColor(reg_disp, cv2.COLOR_RGB2GRAY)
    elif reg_disp.ndim == 3:
        reg_disp = reg_disp[:, :, 0]

    # Handle shape mismatch: crop/pad registered to match reference canvas
    if reg_disp.shape[:2] != (h, w):
        canvas = np.full((h, w), _BORDER_FILL, dtype=np.uint8)
        rh = min(reg_disp.shape[0], h)
        rw = min(reg_disp.shape[1], w)
        canvas[:rh, :rw] = reg_disp[:rh, :rw]
        reg_disp = canvas

    wipe = np.zeros((h, w), dtype=np.uint8)
    wipe[:, :split_x]  = reg_disp[:, :split_x]
    wipe[:, split_x:]  = ref_disp[:, split_x:]

    # Bright cyan-white divider line (2px wide)
    wipe[:, max(0, split_x - 1):min(w, split_x + 1)] = 255
    return wipe
