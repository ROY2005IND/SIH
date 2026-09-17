"""Correspondence filtering and duplicate removal utilities."""
import numpy as np

def filter_by_confidence(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    confidences: np.ndarray,
    min_confidence: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter correspondences by minimum confidence score."""
    if len(confidences) == 0:
        return src_pts, ref_pts, confidences

    keep_mask = confidences >= min_confidence
    return src_pts[keep_mask], ref_pts[keep_mask], confidences[keep_mask]

def filter_duplicates(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    confidences: np.ndarray,
    min_dist_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove redundant co-located keypoints to ensure clean spatial distribution."""
    if len(src_pts) <= 1:
        return src_pts, ref_pts, confidences

    # Sort matches by descending confidence
    sort_idx = np.argsort(-confidences)
    sorted_src = src_pts[sort_idx]
    sorted_ref = ref_pts[sort_idx]
    sorted_conf = confidences[sort_idx]

    kept_indices: list[int] = []
    # Grid hashing for fast duplicate detection
    cell_size = max(min_dist_px, 0.5)
    occupied_cells: set[tuple[int, int]] = set()

    for idx, pt in enumerate(sorted_src):
        cell = (int(pt[0] / cell_size), int(pt[1] / cell_size))
        if cell not in occupied_cells:
            occupied_cells.add(cell)
            kept_indices.append(idx)

    kept_arr = np.array(kept_indices, dtype=int)
    return sorted_src[kept_arr], sorted_ref[kept_arr], sorted_conf[kept_arr]
