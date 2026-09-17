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
    Handles gigapixel rasters safely via spatial tiling to prevent OpenCV OutOfMemoryError.
    """
    h, w = image_u8.shape[:2]
    max_pixels = 9_000_000  # 9 Megapixels threshold for SIFT scale-space memory allocation
    max_dim = 3000

    if h * w > max_pixels or max(h, w) > max_dim:
        tile_size = 2048
        overlap = 128
        stride = tile_size - overlap

        all_pts = []
        all_descs = []
        all_kps = []

        total_tiles_y = (h + stride - 1) // stride
        total_tiles_x = (w + stride - 1) // stride
        num_tiles = max(1, total_tiles_y * total_tiles_x)
        tile_n_features = max(800, n_features // 2)

        sift_tile = cv2.SIFT_create(
            nfeatures=tile_n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

        for y in range(0, h, stride):
            y_end = min(y + tile_size, h)
            for x in range(0, w, stride):
                x_end = min(x + tile_size, w)
                tile_crop = image_u8[y:y_end, x:x_end]

                kps, descs = sift_tile.detectAndCompute(tile_crop, None)
                if kps and descs is not None:
                    for kp, desc in zip(kps, descs):
                        global_x = float(kp.pt[0] + x)
                        global_y = float(kp.pt[1] + y)
                        all_pts.append([global_x, global_y])
                        all_descs.append(desc)

                        new_kp = cv2.KeyPoint(
                            x=global_x,
                            y=global_y,
                            size=kp.size,
                            angle=kp.angle,
                            response=kp.response,
                            octave=kp.octave,
                            class_id=kp.class_id,
                        )
                        all_kps.append(new_kp)

        if not all_pts or len(all_descs) == 0:
            return np.empty((0, 2), dtype=np.float32), None, []

        pts = np.array(all_pts, dtype=np.float32)
        descs = np.vstack(all_descs).astype(np.float32)

        if len(pts) > n_features:
            responses = np.array([kp.response for kp in all_kps])
            top_indices = np.argsort(responses)[::-1][:n_features]
            pts = pts[top_indices]
            descs = descs[top_indices]
            all_kps = [all_kps[i] for i in top_indices]

        return pts, descs, all_kps

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
    h, w = image_u8.shape[:2]
    if h * w > 9_000_000 or max(h, w) > 3000:
        tile_size = 2048
        stride = 1920
        all_pts, all_descs, all_kps = [], [], []
        orb_tile = cv2.ORB_create(nfeatures=max(100, n_features // 10))

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                crop = image_u8[y:min(y+tile_size, h), x:min(x+tile_size, w)]
                kps, descs = orb_tile.detectAndCompute(crop, None)
                if kps and descs is not None:
                    for kp, desc in zip(kps, descs):
                        gx, gy = float(kp.pt[0] + x), float(kp.pt[1] + y)
                        all_pts.append([gx, gy])
                        all_descs.append(desc)
                        all_kps.append(cv2.KeyPoint(x=gx, y=gy, size=kp.size, angle=kp.angle, response=kp.response))

        if not all_pts:
            return np.empty((0, 2), dtype=np.float32), None, []
        return np.array(all_pts, dtype=np.float32), np.vstack(all_descs), all_kps

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
    h, w = image_u8.shape[:2]
    if h * w > 9_000_000 or max(h, w) > 3000:
        # Fallback to tiled SIFT or ORB for massive images to avoid AKAZE nonlinear diffusion crash
        return extract_sift_features(image_u8, n_features=5000)

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
