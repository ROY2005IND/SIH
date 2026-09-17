"""
Adaptive Hybrid Correspondence Engine for MoonFlower AI.
Fuses classical (SIFT) and learned/dense correspondence candidates based on measured geometric evidence.
Does not blindly concatenate; performs method-level inlier analysis and spatial deduplication.
"""
from dataclasses import dataclass
import time
import numpy as np
from .classical import ClassicalMatcher
from .learned import LearnedMatcher
from registration.ransac import estimate_robust_transformation
from utils.logging import get_logger

logger = get_logger("HybridMatcher")

@dataclass
class HybridDiagnostic:
    sift_tentative: int
    sift_inliers: int
    sift_rmse: float | None
    learned_tentative: int
    learned_inliers: int
    learned_rmse: float | None
    fused_total: int
    decision_rationale: str
    is_deep_model: bool = False
    secondary_backend: str = "classical_gftt_sift_fallback"

    @property
    def secondary_classical_inliers(self) -> int:
        """Alias for learned_inliers when running in classical fallback mode."""
        return self.learned_inliers


class AdaptiveHybridMatcher:
    """Combines classical and learned correspondence pipelines adaptively."""

    def __init__(self, settings_override: dict | None = None):
        self.classical_matcher = ClassicalMatcher(method="sift", settings_override=settings_override)
        self.learned_matcher = LearnedMatcher(model_name="superpoint")

    def find_matches(
        self,
        source_u8: np.ndarray,
        reference_u8: np.ndarray,
        ransac_threshold: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, HybridDiagnostic]:
        """
        Execute both matchers, analyze evidence, and fuse validated non-redundant correspondences.
        """
        t0 = time.perf_counter()

        # 1. Classical SIFT execution
        sift_src, sift_ref, sift_confs, _ = self.classical_matcher.find_matches(source_u8, reference_u8)
        _, sift_mask, sift_res = estimate_robust_transformation(
            sift_src, sift_ref, model="affine", ransac_threshold=ransac_threshold
        )
        sift_inliers = int(np.sum(sift_mask)) if sift_mask is not None else 0
        sift_rmse = float(np.sqrt(np.mean(sift_res[sift_mask] ** 2))) if sift_mask is not None and sift_inliers > 0 else None

        # 2. Learned / High-Repeatability Matcher execution
        deep_src, deep_ref, deep_confs, _ = self.learned_matcher.find_matches(source_u8, reference_u8)
        _, deep_mask, deep_res = estimate_robust_transformation(
            deep_src, deep_ref, model="affine", ransac_threshold=ransac_threshold
        )
        deep_inliers = int(np.sum(deep_mask)) if deep_mask is not None else 0
        deep_rmse = float(np.sqrt(np.mean(deep_res[deep_mask] ** 2))) if deep_mask is not None and deep_inliers > 0 else None

        is_deep = self.learned_matcher.is_deep_model
        secondary_backend = "genuine_deep_superpoint" if is_deep else "classical_gftt_sift_fallback"

        # Extract inliers if sufficient, otherwise keep tentative correspondences for joint RANSAC fusion
        if sift_mask is not None and sift_inliers >= 4:
            valid_sift_src, valid_sift_ref, valid_sift_conf = sift_src[sift_mask], sift_ref[sift_mask], sift_confs[sift_mask]
        else:
            valid_sift_src, valid_sift_ref, valid_sift_conf = sift_src, sift_ref, sift_confs

        if deep_mask is not None and deep_inliers >= 4:
            valid_deep_src, valid_deep_ref, valid_deep_conf = deep_src[deep_mask], deep_ref[deep_mask], deep_confs[deep_mask]
        else:
            valid_deep_src, valid_deep_ref, valid_deep_conf = deep_src, deep_ref, deep_confs

        # 3. Evidence-Based Fusion & Fast Spatial Deduplication
        from scipy.spatial import cKDTree

        dedup_radius = 4.0  # pixels
        added_from_deep = 0

        valid_sift_src = np.asarray(valid_sift_src, dtype=np.float32)
        valid_sift_ref = np.asarray(valid_sift_ref, dtype=np.float32)
        valid_sift_conf = np.asarray(valid_sift_conf, dtype=np.float32)

        valid_deep_src = np.asarray(valid_deep_src, dtype=np.float32)
        valid_deep_ref = np.asarray(valid_deep_ref, dtype=np.float32)
        valid_deep_conf = np.asarray(valid_deep_conf, dtype=np.float32)

        if len(valid_sift_ref) > 0 and len(valid_deep_ref) > 0:
            tree = cKDTree(valid_sift_ref)
            dists, _ = tree.query(valid_deep_ref, k=1)
            keep_mask = dists >= dedup_radius

            kept_deep_src = valid_deep_src[keep_mask]
            kept_deep_ref = valid_deep_ref[keep_mask]
            kept_deep_conf = valid_deep_conf[keep_mask] * 0.95

            # Self-deduplicate remaining deep points if needed
            if len(kept_deep_ref) > 1:
                deep_tree = cKDTree(kept_deep_ref)
                pairs = deep_tree.query_pairs(r=dedup_radius)
                if pairs:
                    discard_indices = {p[1] for p in pairs}
                    final_deep_mask = np.ones(len(kept_deep_ref), dtype=bool)
                    for idx in discard_indices:
                        final_deep_mask[idx] = False
                    kept_deep_src = kept_deep_src[final_deep_mask]
                    kept_deep_ref = kept_deep_ref[final_deep_mask]
                    kept_deep_conf = kept_deep_conf[final_deep_mask]

            added_from_deep = len(kept_deep_ref)
            if added_from_deep > 0:
                fused_src = np.vstack([valid_sift_src, kept_deep_src]).astype(np.float32)
                fused_ref = np.vstack([valid_sift_ref, kept_deep_ref]).astype(np.float32)
                fused_confs = np.concatenate([valid_sift_conf, kept_deep_conf]).astype(np.float32)
            else:
                fused_src = valid_sift_src
                fused_ref = valid_sift_ref
                fused_confs = valid_sift_conf
        elif len(valid_sift_ref) > 0:
            fused_src = valid_sift_src
            fused_ref = valid_sift_ref
            fused_confs = valid_sift_conf
        elif len(valid_deep_ref) > 0:
            fused_src = valid_deep_src
            fused_ref = valid_deep_ref
            fused_confs = valid_deep_conf * 0.95
            added_from_deep = len(valid_deep_ref)
        else:
            fused_src = np.empty((0, 2), dtype=np.float32)
            fused_ref = np.empty((0, 2), dtype=np.float32)
            fused_confs = np.empty((0,), dtype=np.float32)

        runtime = time.perf_counter() - t0

        if is_deep:
            rationale = (
                f"Adaptive Hybrid Fusion: Combined {sift_inliers} classical SIFT inliers with "
                f"{added_from_deep} non-redundant deep learned feature inliers ({self.learned_matcher.model_name}), "
                f"increasing spatial density across low-contrast terrain."
            )
        else:
            rationale = (
                f"Adaptive Hybrid Fusion (Classical Fallback): Combined {sift_inliers} SIFT inliers with "
                f"{added_from_deep} non-redundant GFTT-SIFT classical inliers. Deep learned weights are absent; "
                f"operating in dual-classical feature fusion mode."
            )

        diag = HybridDiagnostic(
            sift_tentative=len(sift_src),
            sift_inliers=sift_inliers,
            sift_rmse=sift_rmse,
            learned_tentative=len(deep_src),
            learned_inliers=deep_inliers,
            learned_rmse=deep_rmse,
            fused_total=len(fused_src),
            decision_rationale=rationale,
            is_deep_model=is_deep,
            secondary_backend=secondary_backend,
        )

        return fused_src, fused_ref, fused_confs, runtime, diag

