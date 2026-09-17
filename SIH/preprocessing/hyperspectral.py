"""
Hyperspectral Data Handling Module for Chandrayaan-2 IIRS (250 bands).
Supports band extraction, continuum averaging, and PCA dimensionality reduction.
"""
from typing import Literal
import numpy as np
from sklearn.decomposition import PCA
from .normalization import percentile_stretch

HyperspectralMode = Literal["band", "average", "pca_single", "pca_rgb"]

class HyperspectralProcessor:
    """Processes 3D hyperspectral lunar cubes into registered 2D/3D representations."""

    def __init__(self, cube: np.ndarray, wavelengths: list[float] | None = None):
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D hyperspectral cube (H, W, B), got shape {cube.shape}")
        self.cube = cube
        self.height, self.width, self.n_bands = cube.shape
        self.wavelengths = wavelengths or [round(0.8 + (i * 4.2 / max(self.n_bands - 1, 1)), 3) for i in range(self.n_bands)]

    def extract_band(self, band_index: int) -> np.ndarray:
        """Extract a single spectral band as 8-bit uint8."""
        idx = max(0, min(band_index, self.n_bands - 1))
        band_raw = self.cube[:, :, idx]
        return percentile_stretch(band_raw)

    def extract_band_by_wavelength(self, target_um: float) -> tuple[np.ndarray, int, float]:
        """Extract the band closest to target wavelength in micrometers."""
        diffs = [abs(w - target_um) for w in self.wavelengths]
        best_idx = int(np.argmin(diffs))
        actual_um = self.wavelengths[best_idx]
        return self.extract_band(best_idx), best_idx, actual_um

    def average_bands(self, start_band: int = 0, end_band: int | None = None) -> np.ndarray:
        """Compute continuum band average across a spectral interval."""
        end = end_band or self.n_bands
        sub_cube = self.cube[:, :, start_band:end].astype(np.float32)
        mean_band = np.nanmean(sub_cube, axis=2)
        return percentile_stretch(mean_band)

    def compute_pca(self, n_components: int = 3) -> tuple[np.ndarray, list[float]]:
        """
        Compute PCA across the spectral dimension.
        Returns:
            projected: (H, W, n_components) float32
            explained_variance_ratios: list of variance explained per component.
        """
        flat = self.cube.reshape(-1, self.n_bands).astype(np.float32)
        # Impute NaNs with median if any
        col_means = np.nanmedian(flat, axis=0)
        nan_mask = np.isnan(flat)
        flat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        pca = PCA(n_components=min(n_components, self.n_bands))
        transformed = pca.fit_transform(flat)
        projected = transformed.reshape(self.height, self.width, -1)
        explained = [float(v) for v in pca.explained_variance_ratio_]
        return projected, explained

    def get_principal_component(self, pc_index: int = 0) -> np.ndarray:
        """Extract a specific principal component as 8-bit uint8 (PC1 represents dominant albedo)."""
        projected, _ = self.compute_pca(n_components=max(pc_index + 1, 3))
        comp = projected[:, :, pc_index]
        return percentile_stretch(comp)
