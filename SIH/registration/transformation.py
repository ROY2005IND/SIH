"""
Mathematical transformation models and coordinate mapping functions for lunar imagery.
Supports Translation (2 DoF), Similarity (4 DoF), Affine (6 DoF), and Homography (8 DoF).
"""
from typing import Literal
import numpy as np

TransformationModel = Literal["translation", "similarity", "affine", "homography"]

MODEL_DOF: dict[TransformationModel, int] = {
    "translation": 2,
    "similarity": 4,
    "affine": 6,
    "homography": 8,
}

def apply_transformation(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Transform (N, 2) 2D coordinates using transformation matrix M.
    Supports both 2x3 affine/similarity matrices and 3x3 homography matrices.
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = points.astype(np.float64)
    if M.shape == (2, 3):
        mapped = (pts @ M[:, :2].T) + M[:, 2]
        return mapped.astype(np.float32)
    elif M.shape == (3, 3):
        homog = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        mapped = (homog @ M.T)
        z = mapped[:, 2:3]
        # Prevent division by zero
        z = np.where(np.abs(z) < 1e-8, 1e-8, z)
        return (mapped[:, :2] / z).astype(np.float32)
    else:
        raise ValueError(f"Invalid transformation matrix shape: {M.shape}")

def invert_transformation(M: np.ndarray) -> np.ndarray:
    """Compute the inverse transformation matrix."""
    if M.shape == (2, 3):
        M_3x3 = np.vstack([M, [0.0, 0.0, 1.0]])
        inv_3x3 = np.linalg.inv(M_3x3)
        return inv_3x3[:2, :]
    elif M.shape == (3, 3):
        return np.linalg.inv(M)
    else:
        raise ValueError(f"Invalid transformation matrix shape: {M.shape}")

def calculate_residuals(
    source_pts: np.ndarray,
    ref_pts: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    Calculate Euclidean geometric transfer error in pixels for each match:
    residual_i = || p_ref,i - M(p_src,i) ||
    """
    if len(source_pts) == 0 or len(ref_pts) == 0:
        return np.empty((0,), dtype=np.float32)

    mapped_src = apply_transformation(source_pts, M)
    diffs = ref_pts - mapped_src
    residuals = np.linalg.norm(diffs, axis=1)
    return residuals.astype(np.float32)
