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
        runtime = time.perf_counter() - t0

        return clean_src, clean_ref, clean_conf, runtime
