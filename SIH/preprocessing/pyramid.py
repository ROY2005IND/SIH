"""
Multi-Scale Gaussian Scale Pyramid for Coarse-to-Fine Correspondence Search.
Enables robust matching across massive scale/resolution disparities (e.g. OHRC vs TMC-2).
"""
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class PyramidLevel:
    """A single octave level within the multi-scale Gaussian pyramid."""
    level: int
    scale_factor: float  # e.g. 1.0, 0.5, 0.25
    image: np.ndarray
    width: int
    height: int

class ScalePyramid:
    """Multi-scale Gaussian pyramid builder and coordinate transformer."""

    def __init__(self, base_image_u8: np.ndarray, n_levels: int = 3):
        self.base_image = base_image_u8
        self.n_levels = max(1, n_levels)
        self.levels: list[PyramidLevel] = []
        self._build_pyramid()

    def _build_pyramid(self):
        curr = self.base_image
        scale = 1.0
        for lvl in range(self.n_levels):
            h, w = curr.shape[:2]
            self.levels.append(PyramidLevel(level=lvl, scale_factor=scale, image=curr, width=w, height=h))
            if lvl < self.n_levels - 1:
                curr = cv2.pyrDown(curr)
                scale *= 0.5

    def get_level(self, level: int) -> PyramidLevel:
        """Get pyramid level clamped within [0, n_levels - 1]."""
        idx = max(0, min(level, self.n_levels - 1))
        return self.levels[idx]

    def to_native_coordinates(self, pts_at_level: np.ndarray, level: int) -> np.ndarray:
        """
        Map coordinates from pyramid level l back to native (level 0) coordinate frame:
        P_native = P_level / scale_factor
        """
        if len(pts_at_level) == 0:
            return np.empty((0, 2), dtype=np.float32)
        scale = self.levels[level].scale_factor
        return (pts_at_level / scale).astype(np.float32)

    def get_search_window(
        self,
        coarse_pt: tuple[float, float],
        coarse_level: int,
        target_level: int,
        radius_target_px: int = 32,
    ) -> tuple[int, int, int, int]:
        """
        Calculate local search bounding box [xmin, ymin, xmax, ymax] at target_level
        around coarse_pt mapped from coarse_level.
        """
        cx, cy = coarse_pt
        scale_ratio = self.levels[coarse_level].scale_factor / self.levels[target_level].scale_factor
        tx, ty = cx * scale_ratio, cy * scale_ratio

        target_img = self.levels[target_level]
        xmin = max(0, int(tx - radius_target_px))
        ymin = max(0, int(ty - radius_target_px))
        xmax = min(target_img.width, int(tx + radius_target_px))
        ymax = min(target_img.height, int(ty + radius_target_px))

        return xmin, ymin, xmax, ymax
