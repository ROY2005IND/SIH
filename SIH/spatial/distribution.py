"""
Adaptive Spatial Match Balancer.
Directly implements ISRO's requirement for spatially uniform correspondence distribution.
Prevents clustering on high-contrast crater rims while retaining reliable points across smooth maria.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass
class SpatialBalancingResult:
    """Encapsulates raw vs spatially balanced correspondence sets and uniformity metrics."""
    balanced_indices: np.ndarray
    balanced_source_points: np.ndarray
    balanced_reference_points: np.ndarray
    raw_uniformity_score: float        # [0.0, 100.0]
    balanced_uniformity_score: float   # [0.0, 100.0]
    grid_occupancy_pct: float          # % of cells with at least 1 verified match
    cell_counts_matrix: np.ndarray     # (K, K) point density grid
    grid_size: int                     # K (e.g. 4, 6, 8)

def compute_spatial_uniformity(
    points: np.ndarray,
    image_shape: tuple[int, int],
    grid_size: int = 6,
) -> tuple[float, float, np.ndarray]:
    """
    Calculate documented Spatial Uniformity Score based on normalized grid entropy and occupancy:
        H_norm = -sum(p_i * ln(p_i)) / ln(K^2)
        Uniformity = Occupancy_ratio * H_norm * 100.0
    Returns:
        uniformity_score: float in [0.0, 100.0]
        occupancy_pct: float in [0.0, 100.0]
        cell_matrix: (grid_size, grid_size) integer count array
    """
    if len(points) == 0:
        return 0.0, 0.0, np.zeros((grid_size, grid_size), dtype=int)

    h, w = image_shape[:2]
    K = grid_size
    cell_w = max(float(w) / K, 1e-4)
    cell_h = max(float(h) / K, 1e-4)

    counts = np.zeros((K, K), dtype=int)
    cols = np.clip((points[:, 0] / cell_w).astype(int), 0, K - 1)
    rows = np.clip((points[:, 1] / cell_h).astype(int), 0, K - 1)

    for r, c in zip(rows, cols):
        counts[r, c] += 1

    total_cells = K * K
    occupied_cells = int(np.sum(counts > 0))
    occupancy_ratio = occupied_cells / total_cells

    total_pts = len(points)
    probs = counts[counts > 0].astype(np.float64) / total_pts
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(total_cells)

    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0
    uniformity_score = round(float(occupancy_ratio * norm_entropy * 100.0), 2)
    occupancy_pct = round(occupancy_ratio * 100.0, 2)

    return uniformity_score, occupancy_pct, counts

class AdaptiveSpatialBalancer:
    """Adaptive grid-based correspondence balancer with confidence and distance ranking."""

    def __init__(
        self,
        grid_size: int = 6,
        max_matches_per_cell: int = 20,
        min_intra_cell_dist_px: float = 8.0,
    ):
        self.grid_size = grid_size
        self.max_matches = max_matches_per_cell
        self.min_dist = min_intra_cell_dist_px

    def balance(
        self,
        source_points: np.ndarray,
        reference_points: np.ndarray,
        confidences: np.ndarray,
        image_shape: tuple[int, int],
        residuals: np.ndarray | None = None,
    ) -> SpatialBalancingResult:
        """
        Prune over-dense clusters and retain well-distributed, high-confidence points.
        """
        n_pts = len(source_points)
        if n_pts == 0:
            empty_idx = np.empty((0,), dtype=int)
            empty_pts = np.empty((0, 2), dtype=np.float32)
            counts = np.zeros((self.grid_size, self.grid_size), dtype=int)
            return SpatialBalancingResult(
                balanced_indices=empty_idx,
                balanced_source_points=empty_pts,
                balanced_reference_points=empty_pts,
                raw_uniformity_score=0.0,
                balanced_uniformity_score=0.0,
                grid_occupancy_pct=0.0,
                cell_counts_matrix=counts,
                grid_size=self.grid_size,
            )

        # 1. Compute baseline raw uniformity
        raw_uniformity, _, _ = compute_spatial_uniformity(reference_points, image_shape, self.grid_size)

        h, w = image_shape[:2]
        K = self.grid_size
        cell_w = max(float(w) / K, 1e-4)
        cell_h = max(float(h) / K, 1e-4)

        # Composite score: reward high confidence, penalize high residual
        if residuals is not None and len(residuals) == n_pts:
            norm_res = np.clip(residuals / (np.median(residuals) + 1e-5), 0.0, 3.0)
            composite_score = confidences - 0.25 * norm_res
        else:
            composite_score = confidences.copy()

        # Group indices by cell
        cols = np.clip((reference_points[:, 0] / cell_w).astype(int), 0, K - 1)
        rows = np.clip((reference_points[:, 1] / cell_h).astype(int), 0, K - 1)

        cell_bins: dict[tuple[int, int], list[int]] = {(r, c): [] for r in range(K) for c in range(K)}
        for idx in range(n_pts):
            cell_bins[(rows[idx], cols[idx])].append(idx)

        selected_indices: list[int] = []

        # 2. Select balanced subset per cell
        for (r, c), indices in cell_bins.items():
            if not indices:
                continue

            # Rank by descending composite score
            ranked = sorted(indices, key=lambda i: composite_score[i], reverse=True)

            cell_selected: list[int] = []
            cell_pts: list[np.ndarray] = []

            for cand_idx in ranked:
                cand_pt = reference_points[cand_idx]

                # Intra-cell spacing constraint
                if cell_pts:
                    dists = [np.linalg.norm(cand_pt - p) for p in cell_pts]
                    if min(dists) < self.min_dist:
                        continue

                cell_selected.append(cand_idx)
                cell_pts.append(cand_pt)

                if len(cell_selected) >= self.max_matches:
                    break

            selected_indices.extend(cell_selected)

        balanced_idx_arr = np.array(sorted(selected_indices), dtype=int)
        balanced_src = source_points[balanced_idx_arr]
        balanced_ref = reference_points[balanced_idx_arr]

        # 3. Compute post-balancing uniformity and density matrix
        balanced_uniformity, occupancy_pct, cell_matrix = compute_spatial_uniformity(
            balanced_ref, image_shape, self.grid_size
        )

        return SpatialBalancingResult(
            balanced_indices=balanced_idx_arr,
            balanced_source_points=balanced_src,
            balanced_reference_points=balanced_ref,
            raw_uniformity_score=raw_uniformity,
            balanced_uniformity_score=balanced_uniformity,
            grid_occupancy_pct=occupancy_pct,
            cell_counts_matrix=cell_matrix,
            grid_size=self.grid_size,
        )
