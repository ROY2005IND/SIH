"""
Classical Keypoint & Descriptor Extractors (SIFT and AKAZE).
Provides scale and rotation-invariant feature detection for lunar surface topography.
"""
from typing import Literal
import numpy as np
import cv2
from config import load_settings

FeatureType = Literal["sift", "akaze"]

def extract_sift_features(
    image_u8: np.ndarray,
    n_features: int = 5000,
    contrast_threshold: float = 0.03,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
) -> tuple[np.ndarray, np.ndarray | None, list[cv2.KeyPoint]]:
    """
    Extract Scale-Invariant Feature Transform (SIFT) keypoints and 128-D descriptors.
    Returns:
        points: (N, 2) float32 coordinates [x, y]
        descriptors: (N, 128) float32 descriptor matrix
        cv_kps: list of OpenCV KeyPoint objects
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    cv_kps, descs = sift.detectAndCompute(image_u8, None)
    if not cv_kps or descs is None:
        return np.empty((0, 2), dtype=np.float32), None, []

    pts = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
    return pts, descs.astype(np.float32), list(cv_kps)

def extract_orb_features(
    image_u8: np.ndarray,
    n_features: int = 3000,
) -> tuple[np.ndarray, np.ndarray | None, list[cv2.KeyPoint]]:
    """
    Extract Oriented FAST and Rotated BRIEF (ORB) binary keypoints.
    Computationally lightweight binary alternative to SIFT.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    cv_kps, descs = orb.detectAndCompute(image_u8, None)
    if not cv_kps or descs is None:
        return np.empty((0, 2), dtype=np.float32), None, []

    pts = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
    return pts, descs, list(cv_kps)

def extract_akaze_features(
    image_u8: np.ndarray,
    threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray | None, list[cv2.KeyPoint]]:
    """
    Extract Accelerated-KAZE (AKAZE) keypoints using non-linear scale space filtering.
    Falls back gracefully to ORB if AKAZE is not compiled into the local OpenCV distribution.
    """
    if hasattr(cv2, "AKAZE_create"):
        akaze = cv2.AKAZE_create(threshold=threshold)
        cv_kps, descs = akaze.detectAndCompute(image_u8, None)
        if not cv_kps or descs is None:
            return np.empty((0, 2), dtype=np.float32), None, []
        pts = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
        return pts, descs, list(cv_kps)
    elif hasattr(cv2, "AKAZE"):
        akaze = cv2.AKAZE.create(threshold=threshold)
        cv_kps, descs = akaze.detectAndCompute(image_u8, None)
        if not cv_kps or descs is None:
            return np.empty((0, 2), dtype=np.float32), None, []
        pts = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
        return pts, descs, list(cv_kps)
    else:
        # Graceful fallback to ORB binary descriptor
        return extract_orb_features(image_u8, n_features=3000)

class ClassicalFeatureExtractor:
    """Configurable classical feature extraction interface."""

    def __init__(self, method: FeatureType = "sift", config_override: dict | None = None):
        self.method = method.lower()
        settings = config_override or load_settings()
        feat_cfg = settings.get("features", {})
        self.sift_cfg = feat_cfg.get("sift", {})
        self.akaze_cfg = feat_cfg.get("akaze", {})

    def extract(self, image_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, list[cv2.KeyPoint]]:
        """Extract keypoints and descriptors using configured algorithm."""
        if self.method == "akaze":
            thresh = self.akaze_cfg.get("threshold", 0.001)
            return extract_akaze_features(image_u8, threshold=thresh)
        else:  # default to SIFT
            return extract_sift_features(
                image_u8,
                n_features=self.sift_cfg.get("n_features", 5000),
                contrast_threshold=self.sift_cfg.get("contrast_threshold", 0.03),
                edge_threshold=self.sift_cfg.get("edge_threshold", 10.0),
                sigma=self.sift_cfg.get("sigma", 1.6),
            )
