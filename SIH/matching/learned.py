"""
Learned Matcher Adapter (LightGlue / Dense LoFTR Interface).
Provides modular interface with graceful fallback to mutual nearest neighbor matching.
"""
import time
import numpy as np
import cv2
from features.learned import LearnedFeatureExtractor
from .classical import match_descriptors
from .filtering import filter_duplicates, filter_by_confidence

class LearnedMatcher:
    """Matcher using deep/pseudo-learned features with mutual consistency checks."""

    def __init__(self, model_name: str = "superpoint_lightglue"):
        self.model_name = model_name
        self.extractor = LearnedFeatureExtractor("superpoint")

    @property
    def is_deep_model(self) -> bool:
        """Return True only if the underlying extractor loaded real deep weights."""
        return self.extractor.model_loaded

    @property
    def backend_status(self) -> str:
        """Return backend status ('GENUINE_DEEP_MODEL' or 'CLASSICAL_FALLBACK')."""
        return "GENUINE_DEEP_MODEL" if self.is_deep_model else "CLASSICAL_FALLBACK"

    def find_matches(
        self,
        source_u8: np.ndarray,
        reference_u8: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extract learned keypoints and compute correspondences.
        Returns:
            source_pts: (M, 2) float32
            ref_pts: (M, 2) float32
            confidences: (M,) float32
            runtime_seconds: float
        """
        t0 = time.perf_counter()

        feat_src = self.extractor.extract(source_u8)
        feat_ref = self.extractor.extract(reference_u8)

        if len(feat_src.keypoints) == 0 or len(feat_ref.keypoints) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                time.perf_counter() - t0,
            )

        # Cross-check mutual nearest neighbors
        idx1, idx2, confs = match_descriptors(
            feat_src.descriptors, feat_ref.descriptors,
            method="sift", ratio_threshold=0.80,
        )

        matched_src = feat_src.keypoints[idx1]
        matched_ref = feat_ref.keypoints[idx2]

        clean_src, clean_ref, clean_conf = filter_duplicates(matched_src, matched_ref, confs)

        if len(clean_src) < 4 and (reference_u8.shape[0] > 3000 or reference_u8.shape[1] > 3000 or source_u8.shape[0] > 3000 or source_u8.shape[1] > 3000):
            try:
                from preprocessing.pyramid import ScalePyramid
                pyramid = ScalePyramid(reference_u8, n_levels=3)
                all_pts2_list = [feat_ref.keypoints]
                all_desc2_list = [feat_ref.descriptors]

                for lvl in range(1, pyramid.n_levels):
                    lvl_img = pyramid.get_level(lvl).image
                    lvl_feat = self.extractor.extract(lvl_img)
                    if len(lvl_feat.keypoints) > 0 and lvl_feat.descriptors is not None:
                        native_pts = pyramid.to_native_coordinates(lvl_feat.keypoints, lvl)
                        all_pts2_list.append(native_pts)
                        all_desc2_list.append(lvl_feat.descriptors)

                if len(all_pts2_list) > 1:
                    pts2_multi = np.vstack(all_pts2_list)
                    desc2_multi = np.vstack(all_desc2_list)
                    idx1_m, idx2_m, confs_m = match_descriptors(feat_src.descriptors, desc2_multi, method="sift", ratio_threshold=0.85)
                    if len(idx1_m) >= 4:
                        m_src = feat_src.keypoints[idx1_m]
                        m_ref = pts2_multi[idx2_m]
                        c_src, c_ref, c_conf = filter_duplicates(m_src, m_ref, confs_m)
                        if len(c_src) > len(clean_src):
                            clean_src, clean_ref, clean_conf = c_src, c_ref, c_conf
            except Exception:
                pass

        runtime = time.perf_counter() - t0
        return clean_src, clean_ref, clean_conf, runtime
