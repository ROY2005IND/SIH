"""
Local Non-Rigid Warping and Spatial Residual Strain Analysis Module.
Provides Piecewise Affine and Thin Plate Spline (TPS) warping only when justified by structured spatial residual strain.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.spatial import Delaunay
from .transformation import TransformationModel, calculate_residuals

@dataclass
class ResidualStrainReport:
    is_strain_structured: bool
    spatial_autocorrelation: float   # Moran's I approximation
    mean_local_variance: float
    recommendation: str

def analyze_residual_spatial_strain(
    points: np.ndarray,
    residuals: np.ndarray,
    grid_size: int = 4,
) -> ResidualStrainReport:
    """
    Evaluate whether residual errors are spatially autocorrelated (indicating non-planar topographic terrain relief or unmodeled lens distortion)
    versus uniform Gaussian noise.
    """
    if len(points) < 8 or len(residuals) != len(points):
        return ResidualStrainReport(
            is_strain_structured=False,
            spatial_autocorrelation=0.0,
            mean_local_variance=0.0,
            recommendation="Insufficient points to determine spatial residual strain. Retain global linear model.",
        )

    # Compute spatial Moran's I approximation
    res_mean = np.mean(residuals)
    res_dev = residuals - res_mean
    variance = np.sum(res_dev ** 2)

    if variance < 1e-6:
        return ResidualStrainReport(
            is_strain_structured=False,
            spatial_autocorrelation=0.0,
            mean_local_variance=0.0,
            recommendation="Zero residual variance. Global model is mathematically exact.",
        )

    # Distance-based spatial weight matrix W
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    # K-nearest neighbors weights
    k_neighbors = min(4, len(points) - 1)
    W = np.zeros_like(dists)
    for i in range(len(points)):
        nn_idx = np.argsort(dists[i])[:k_neighbors]
        W[i, nn_idx] = 1.0

    w_sum = np.sum(W)
    n = len(points)
    numerator = np.sum(W * np.outer(res_dev, res_dev))
    moran_i = float((n / max(w_sum, 1e-6)) * (numerator / variance))

    # Moran's I > 0.25 indicates significant spatial clustering of errors
    is_structured = moran_i > 0.25 and np.max(residuals) > 2.0
    if is_structured:
        rec = f"Spatial residual strain detected (Moran's I={moran_i:.2f} > 0.25). Local topographic parallax justifies Piecewise Affine warping."
    else:
        rec = f"Residuals are spatially uncorrelated (Moran's I={moran_i:.2f} <= 0.25). Global linear model is scientifically optimal."

    return ResidualStrainReport(
        is_strain_structured=is_structured,
        spatial_autocorrelation=round(moran_i, 3),
        mean_local_variance=round(float(variance / n), 3),
        recommendation=rec,
    )

def warp_piecewise_affine(
    source_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    reference_shape: tuple[int, int],
) -> np.ndarray:
    """
    Perform local piecewise affine warping using Delaunay triangulation.
    Maps each triangle in the source image to the corresponding triangle in the reference image.
    """
    ref_h, ref_w = reference_shape[:2]
    out_img = np.zeros((ref_h, ref_w), dtype=np.uint8)

    if len(reference_points) < 4:
        return source_image

    # Add corner anchor points to ensure full canvas coverage
    anchors_ref = np.array([[0, 0], [ref_w - 1, 0], [0, ref_h - 1], [ref_w - 1, ref_h - 1]], dtype=np.float32)
    # Estimate global affine to place anchor points in source frame
    M_global, _ = cv2.estimateAffine2D(reference_points, source_points)
    if M_global is None:
        anchors_src = anchors_ref.copy()
    else:
        anchors_src = (anchors_ref @ M_global[:, :2].T) + M_global[:, 2]

    all_ref = np.vstack([reference_points, anchors_ref])
    all_src = np.vstack([source_points, anchors_src])

    tri = Delaunay(all_ref)

    for simplex in tri.simplices:
        tri_ref = all_ref[simplex].astype(np.float32)
        tri_src = all_src[simplex].astype(np.float32)

        # Bounding box of reference triangle
        rx_min, ry_min = np.floor(np.min(tri_ref, axis=0)).astype(int)
        rx_max, ry_max = np.ceil(np.max(tri_ref, axis=0)).astype(int)

        rx_min = max(0, rx_min)
        ry_min = max(0, ry_min)
        rx_max = min(ref_w, rx_max + 1)
        ry_max = min(ref_h, ry_max + 1)

        bw = rx_max - rx_min
        bh = ry_max - ry_min
        if bw <= 0 or bh <= 0:
            continue

        # Triangle coordinates relative to bounding box
        tri_ref_offset = tri_ref.copy()
        tri_ref_offset[:, 0] -= rx_min
        tri_ref_offset[:, 1] -= ry_min

        # Compute affine transform from source triangle to reference triangle
        M_tri = cv2.getAffineTransform(tri_src, tri_ref_offset)

        # Warp source patch into reference bounding box
        warped_patch = cv2.warpAffine(
            source_image, M_tri, (bw, bh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Mask triangle area
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_ref_offset.astype(np.int32), 255)

        # Paste into output
        roi = out_img[ry_min:ry_max, rx_min:rx_max]
        np.copyto(roi, warped_patch, where=(mask == 255))

    return out_img
