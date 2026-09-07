"""
Planetary Correspondence Visualization Engine.
Renders side-by-side correspondence rays with distinct inlier (green) vs outlier (red) styling.
"""
import numpy as np
import cv2

def draw_correspondence_rays(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_draw: int = 200,
) -> np.ndarray:
    """
    Render dual-image side-by-side panel with correspondence vector rays.
    Green rays = geometrically verified inliers.
    Red rays = rejected outlier matches.
    """
    h1, w1 = img_src.shape[:2]
    h2, w2 = img_ref.shape[:2]

    # Target canvas
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2

    # Prepare 3-channel RGB canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    src_rgb = cv2.cvtColor(img_src, cv2.COLOR_GRAY2RGB) if img_src.ndim == 2 else img_src
    ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB) if img_ref.ndim == 2 else img_ref

    canvas[:h1, :w1] = src_rgb
    canvas[:h2, w1:w1 + w2] = ref_rgb

    # Boundary separator
    cv2.line(canvas, (w1, 0), (w1, canvas_h), (30, 41, 59), 2)

    n_pts = len(src_pts)
    if n_pts == 0:
        return canvas

    mask = inlier_mask if inlier_mask is not None else np.ones(n_pts, dtype=bool)

    # Subsample if matches exceed max_draw
    if n_pts > max_draw:
        indices = np.linspace(0, n_pts - 1, max_draw, dtype=int)
    else:
        indices = np.arange(n_pts)

    # Draw outliers first (red, lower visual hierarchy)
    for idx in indices:
        if not mask[idx]:
            pt1 = (int(round(src_pts[idx, 0])), int(round(src_pts[idx, 1])))
            pt2 = (int(round(ref_pts[idx, 0] + w1)), int(round(ref_pts[idx, 1])))
            cv2.line(canvas, pt1, pt2, (239, 68, 68), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 2, (239, 68, 68), -1)
            cv2.circle(canvas, pt2, 2, (239, 68, 68), -1)

    # Draw inliers second (bright green, higher visual hierarchy)
    for idx in indices:
        if mask[idx]:
            pt1 = (int(round(src_pts[idx, 0])), int(round(src_pts[idx, 1])))
            pt2 = (int(round(ref_pts[idx, 0] + w1)), int(round(ref_pts[idx, 1])))
            cv2.line(canvas, pt1, pt2, (34, 197, 94), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (56, 189, 248), -1)
            cv2.circle(canvas, pt2, 3, (56, 189, 248), -1)

    return canvas

def draw_inlier_outlier_scatter(
    src_pts: np.ndarray,
    inlier_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate coordinates into inlier and outlier arrays."""
    if inlier_mask is None or len(src_pts) == 0:
        return src_pts, np.empty((0, 2), dtype=np.float32)
    return src_pts[inlier_mask], src_pts[~inlier_mask]
