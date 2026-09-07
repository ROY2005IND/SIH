"""
Robust Geometric Estimation via RANSAC / USAC_MAGSAC.
Estimates transformation matrices while rejecting severe outlier correspondences.
"""
import numpy as np
import cv2
from .transformation import TransformationModel, calculate_residuals

def estimate_robust_transformation(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    model: TransformationModel = "affine",
    ransac_threshold: float = 3.0,
    max_iters: int = 3000,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Robustly estimate geometric transformation between corresponding points.
    Returns:
        M: transformation matrix (2x3 or 3x3) or None if estimation failed
        inlier_mask: (N,) bool array or None
        residuals: (N,) float32 Euclidean pixel transfer errors or None
    """
    n_pts = len(source_points)
    min_pts_required = {"translation": 2, "similarity": 3, "affine": 3, "homography": 4}

    if n_pts < min_pts_required.get(model, 3):
        return None, None, None

    src = source_points.astype(np.float32)
    ref = reference_points.astype(np.float32)

    # For affine and similarity, OpenCV requires cv2.RANSAC or cv2.LMEDS
    affine_ransac_flag = cv2.RANSAC
    # For homography, USAC_MAGSAC provides state-of-the-art robust estimation
    homog_ransac_flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC

    M: np.ndarray | None = None
    mask_arr: np.ndarray | None = None

    if model == "translation":
        # Compute median delta [dx, dy] with RANSAC-like thresholding
        deltas = ref - src
        med_delta = np.median(deltas, axis=0)
        res = np.linalg.norm(deltas - med_delta, axis=1)
        mask = res <= ransac_threshold
        if np.sum(mask) >= 2:
            refined_delta = np.mean(deltas[mask], axis=0)
            M = np.array([[1.0, 0.0, refined_delta[0]], [0.0, 1.0, refined_delta[1]]], dtype=np.float64)
            mask_arr = mask

    elif model == "similarity":
        M, mask = cv2.estimateAffinePartial2D(
            src, ref,
            method=affine_ransac_flag,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is not None:
            mask_arr = mask.ravel().astype(bool)

    elif model == "affine":
        M, mask = cv2.estimateAffine2D(
            src, ref,
            method=affine_ransac_flag,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is not None:
            mask_arr = mask.ravel().astype(bool)

    elif model == "homography":
        M, mask = cv2.findHomography(
            src, ref,
            method=homog_ransac_flag,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is not None:
            mask_arr = mask.ravel().astype(bool)

    if M is None or mask_arr is None or np.sum(mask_arr) < min_pts_required.get(model, 3):
        return None, None, None

    # Calculate actual pixel transfer residuals
    residuals = calculate_residuals(src, ref, M)
    return M, mask_arr, residuals
