"""
Genuine Sub-Pixel Refinement Engine via Local Normalized Cross-Correlation (NCC)
and 2D Quadratic / Parabolic Surface Peak Optimization.
"""
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class SubPixelRefinementResult:
    """Encapsulates sub-pixel correspondence coordinates and displacement measurements."""
    initial_points: np.ndarray       # (N, 2) float32 integer coords
    refined_points: np.ndarray       # (N, 2) float32 fractional coords
    offsets: np.ndarray              # (N, 2) float32 [dx, dy] sub-pixel displacement vectors
    mean_offset_px: float            # Mean magnitude of sub-pixel adjustments
    max_offset_px: float             # Max adjustment
    converged_mask: np.ndarray       # (N,) bool indicating successful peak fitting
    n_converged: int

def _fit_2d_parabolic_peak(patch_3x3: np.ndarray) -> tuple[float, float, bool]:
    """
    Fit 2D parabolic surface f(u, v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
    over 3x3 local grid centered at (0, 0).
    Solves for continuous extremum: grad(f) = 0.
    Returns:
        delta_u, delta_v: fractional offset from grid center
        is_valid: boolean indicating true local maximum with negative curvature
    """
    z = patch_3x3
    # Derivatives via central differences on 3x3 grid
    # d = df/du, e = df/dv
    d = (z[1, 2] - z[1, 0]) / 2.0
    e = (z[2, 1] - z[0, 1]) / 2.0

    # 2a = d^2f/du^2, 2b = d^2f/dv^2
    two_a = z[1, 2] - 2.0 * z[1, 1] + z[1, 0]
    two_b = z[2, 1] - 2.0 * z[1, 1] + z[0, 1]

    # c = d^2f/dudv
    c = (z[2, 2] - z[2, 0] - z[0, 2] + z[0, 0]) / 4.0

    det = two_a * two_b - (c ** 2)

    # Check for true maximum (Hessian negative definite: a < 0, b < 0, det > 0)
    if two_a >= 0 or two_b >= 0 or det <= 1e-7:
        return 0.0, 0.0, False

    # Solve [2a  c; c  2b] * [du; dv] = -[d; e]
    inv_det = 1.0 / det
    delta_u = (-two_b * d + c * e) * inv_det
    delta_v = (-two_a * e + c * d) * inv_det

    # Reject unconditioned solutions outside search grid
    if abs(delta_u) > 1.0 or abs(delta_v) > 1.0:
        return 0.0, 0.0, False

    return float(delta_u), float(delta_v), True

def refine_subpixel_correspondences(
    source_image_u8: np.ndarray,
    reference_image_u8: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    patch_size: int = 11,
    search_radius_px: int = 2,
    max_offset_px: float = 1.5,
) -> SubPixelRefinementResult:
    """
    Perform local sub-pixel refinement around tentative integer correspondences.
    Extracts local template, searches NCC neighborhood, and solves 2D quadratic peak.
    """
    n_pts = len(reference_points)
    if n_pts == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return SubPixelRefinementResult(
            initial_points=empty,
            refined_points=empty,
            offsets=empty,
            mean_offset_px=0.0,
            max_offset_px=0.0,
            converged_mask=np.empty((0,), dtype=bool),
            n_converged=0,
        )

    src_h, src_w = source_image_u8.shape[:2]
    ref_h, ref_w = reference_image_u8.shape[:2]

    refined_pts = reference_points.copy().astype(np.float32)
    offsets = np.zeros((n_pts, 2), dtype=np.float32)
    converged = np.zeros(n_pts, dtype=bool)

    half_p = patch_size // 2
    r_search = search_radius_px

    for idx in range(n_pts):
        sx, sy = int(round(source_points[idx, 0])), int(round(source_points[idx, 1]))
        rx, ry = int(round(reference_points[idx, 0])), int(round(reference_points[idx, 1]))

        # Boundary checks
        if (
            sx - half_p < 0 or sx + half_p >= src_w or
            sy - half_p < 0 or sy + half_p >= src_h or
            rx - half_p - r_search < 0 or rx + half_p + r_search >= ref_w or
            ry - half_p - r_search < 0 or ry + half_p + r_search >= ref_h
        ):
            continue

        template = source_image_u8[sy - half_p:sy + half_p + 1, sx - half_p:sx + half_p + 1]
        search_roi = reference_image_u8[
            ry - half_p - r_search:ry + half_p + r_search + 1,
            rx - half_p - r_search:rx + half_p + r_search + 1,
        ]

        if template.shape != (patch_size, patch_size) or search_roi.shape[0] < patch_size:
            continue

        # Normalized cross-correlation
        ncc_map = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_map)

        peak_u, peak_v = max_loc  # in NCC map coordinates

        # Ensure peak is strictly interior for 3x3 neighborhood extraction
        map_h, map_w = ncc_map.shape
        if peak_u <= 0 or peak_u >= map_w - 1 or peak_v <= 0 or peak_v >= map_h - 1:
            continue

        patch_3x3 = ncc_map[peak_v - 1:peak_v + 2, peak_u - 1:peak_u + 2]
        delta_u, delta_v, is_valid = _fit_2d_parabolic_peak(patch_3x3)

        if is_valid:
            # Discrete displacement from reference initial point
            dx_int = float(peak_u - r_search)
            dy_int = float(peak_v - r_search)

            total_dx = dx_int + delta_u
            total_dy = dy_int + delta_v

            dist = float(np.sqrt(total_dx ** 2 + total_dy ** 2))
            if dist <= max_offset_px:
                refined_pts[idx, 0] = reference_points[idx, 0] + total_dx
                refined_pts[idx, 1] = reference_points[idx, 1] + total_dy
                offsets[idx, 0] = total_dx
                offsets[idx, 1] = total_dy
                converged[idx] = True

    offset_norms = np.linalg.norm(offsets[converged], axis=1) if np.any(converged) else np.array([0.0])
    mean_offset = float(np.mean(offset_norms))
    max_offset = float(np.max(offset_norms))

    return SubPixelRefinementResult(
        initial_points=reference_points,
        refined_points=refined_pts,
        offsets=offsets,
        mean_offset_px=round(mean_offset, 3),
        max_offset_px=round(max_offset, 3),
        converged_mask=converged,
        n_converged=int(np.sum(converged)),
    )
