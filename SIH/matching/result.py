"""Standardized correspondence and registration result contracts."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MatchResult:
    """Standardized result object returned by all MoonFlower correspondence & registration engines."""
    method: str
    source_points: np.ndarray        # (N, 2) float32 [x, y]
    reference_points: np.ndarray     # (N, 2) float32 [x, y]
    confidences: np.ndarray          # (N,) float32 [0.0, 1.0]
    inlier_mask: np.ndarray | None = None   # (N,) bool
    transformation: np.ndarray | None = None# (2, 3) or (3, 3) float64
    transformation_type: str | None = None  # "translation" | "similarity" | "affine" | "homography"
    residuals: np.ndarray | None = None     # (N,) float32 transfer errors in pixels
    runtime_seconds: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_matches(self) -> int:
        """Total tentative matches found."""
        return len(self.source_points)

    @property
    def n_inliers(self) -> int:
        """Count of geometrically verified inliers."""
        if self.inlier_mask is None:
            return 0
        return int(np.sum(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        """Ratio of inliers to tentative matches."""
        if self.n_matches == 0 or self.inlier_mask is None:
            return 0.0
        return float(self.n_inliers / self.n_matches)

    @property
    def inlier_source_points(self) -> np.ndarray:
        """Source points corresponding to inliers."""
        if self.inlier_mask is None or self.n_matches == 0:
            return np.empty((0, 2), dtype=np.float32)
        return self.source_points[self.inlier_mask]

    @property
    def inlier_reference_points(self) -> np.ndarray:
        """Reference points corresponding to inliers."""
        if self.inlier_mask is None or self.n_matches == 0:
            return np.empty((0, 2), dtype=np.float32)
        return self.reference_points[self.inlier_mask]

    @property
    def rmse(self) -> float | None:
        """Root Mean Square Error of inlier residuals."""
        if self.residuals is None or len(self.residuals) == 0:
            return None
        valid_residuals = self.residuals if self.inlier_mask is None else self.residuals[self.inlier_mask]
        if len(valid_residuals) == 0:
            return None
        return float(np.sqrt(np.mean(valid_residuals ** 2)))

    @property
    def median_residual(self) -> float | None:
        """Median residual of inliers (robust against remaining leverage points)."""
        if self.residuals is None or len(self.residuals) == 0:
            return None
        valid_residuals = self.residuals if self.inlier_mask is None else self.residuals[self.inlier_mask]
        if len(valid_residuals) == 0:
            return None
        return float(np.median(valid_residuals))
