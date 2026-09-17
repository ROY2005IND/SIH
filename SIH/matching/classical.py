"""
Classical Descriptor Matcher (FLANN and BFMatcher) with Lowe's Ratio Test.
"""
import time
import numpy as np
import cv2
from features.classical import ClassicalFeatureExtractor
from .filtering import filter_duplicates, filter_by_confidence
from .result import MatchResult
from config import load_settings

def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str = "sift",
    ratio_threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match descriptors using FLANN (for SIFT float32) or BFMatcher (for AKAZE binary).
    Applies Lowe's ratio test (d1 < ratio * d2) and calculates match confidence.
    Returns:
        idx1: indices in desc1
        idx2: indices in desc2
        confidences: (M,) float32 match confidence scores in [0, 1]
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=np.float32)

    if method.lower() == "akaze":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        # FLANN matcher for floating point descriptors (SIFT)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # 2-Nearest Neighbors search
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    conf_list: list[float] = []

    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            # Lowe's distance ratio test
            if m.distance < ratio_threshold * n.distance:
                idx1_list.append(m.queryIdx)
                idx2_list.append(m.trainIdx)
                # Confidence: distance ratio complement
                confidence = float(np.clip(1.0 - (m.distance / (n.distance + 1e-7)), 0.0, 1.0))
                conf_list.append(confidence)

    return (
        np.array(idx1_list, dtype=int),
        np.array(idx2_list, dtype=int),
        np.array(conf_list, dtype=np.float32),
    )

class ClassicalMatcher:
    """End-to-end classical feature extraction and descriptor matching pipeline."""

    def __init__(self, method: str = "sift", settings_override: dict | None = None):
        self.method = method.lower()
        self.settings = settings_override or load_settings()
        self.extractor = ClassicalFeatureExtractor(self.method, self.settings)
        match_cfg = self.settings.get("matching", {})
        self.ratio_test = match_cfg.get("ratio_test", 0.75)
        self.conf_thresh = match_cfg.get("confidence_threshold", 0.50)

    def find_matches(
        self,
        source_u8: np.ndarray,
        reference_u8: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extract features and match descriptors between source and reference images.
        Returns:
            source_pts: (M, 2) float32
            ref_pts: (M, 2) float32
            confidences: (M,) float32
            runtime_seconds: float
        """
        t0 = time.perf_counter()

        pts1, desc1, _ = self.extractor.extract(source_u8)
        pts2, desc2, _ = self.extractor.extract(reference_u8)

        if desc1 is None or desc2 is None or len(pts1) == 0 or len(pts2) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                time.perf_counter() - t0,
            )

        idx1, idx2, confs = match_descriptors(desc1, desc2, method=self.method, ratio_threshold=self.ratio_test)
        if len(idx1) < 10:
            # Progressive ratio relaxation for challenging cross-scale or cross-sensor pairs
            idx1, idx2, confs = match_descriptors(desc1, desc2, method=self.method, ratio_threshold=0.82)

        matched_src = pts1[idx1]
        matched_ref = pts2[idx2]

        # Filter by minimum confidence & remove duplicates
        filtered_src, filtered_ref, filtered_conf = filter_by_confidence(
            matched_src, matched_ref, confs, min_confidence=self.conf_thresh
        )
        if len(filtered_src) < 6 and len(matched_src) >= 4:
            # Relax confidence threshold if initial matches are sparse
            filtered_src, filtered_ref, filtered_conf = matched_src, matched_ref, confs

        clean_src, clean_ref, clean_conf = filter_duplicates(filtered_src, filtered_ref, filtered_conf)

        if len(clean_src) < 4 and (reference_u8.shape[0] > 3000 or reference_u8.shape[1] > 3000 or source_u8.shape[0] > 3000 or source_u8.shape[1] > 3000):
            # Scale-Disparity Recovery: Build a 3-level Gaussian Pyramid for reference image
            try:
                from preprocessing.pyramid import ScalePyramid
                pyramid = ScalePyramid(reference_u8, n_levels=3)

                all_pts2_list = [pts2]
                all_desc2_list = [desc2]

                for lvl in range(1, pyramid.n_levels):
                    lvl_img = pyramid.get_level(lvl).image
                    pts_lvl, desc_lvl, _ = self.extractor.extract(lvl_img)
                    if len(pts_lvl) > 0 and desc_lvl is not None:
                        native_pts = pyramid.to_native_coordinates(pts_lvl, lvl)
                        all_pts2_list.append(native_pts)
                        all_desc2_list.append(desc_lvl)

                if len(all_pts2_list) > 1:
                    pts2_multi = np.vstack(all_pts2_list)
                    desc2_multi = np.vstack(all_desc2_list)

                    idx1_m, idx2_m, confs_m = match_descriptors(desc1, desc2_multi, method=self.method, ratio_threshold=0.85)
                    if len(idx1_m) >= 4:
                        matched_src_m = pts1[idx1_m]
                        matched_ref_m = pts2_multi[idx2_m]
                        c_src, c_ref, c_conf = filter_duplicates(matched_src_m, matched_ref_m, confs_m)
                        if len(c_src) > len(clean_src):
                            clean_src, clean_ref, clean_conf = c_src, c_ref, c_conf
            except Exception:
                pass

        runtime = time.perf_counter() - t0
        return clean_src, clean_ref, clean_conf, runtime
