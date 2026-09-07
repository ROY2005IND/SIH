"""
Controlled Lunar-Like Synthetic Benchmark Generator.
NOTE: Labeled strictly as Simplified Photometric Approximation for controlled testing.
Never represented as real ISRO flight performance.
"""
from typing import Sequence
import numpy as np
import cv2
from metadata.geometry import sun_vector_from_angles

def generate_lunar_terrain(
    size: int = 512,
    seed: int = 42,
    crater_density: str = "moderate",
) -> np.ndarray:
    """
    Generate a 2D digital elevation model (DEM) representing synthetic lunar terrain.
    Combines multi-scale undulating background noise with impact crater bowl and rim profiles.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    # 1. Base undulating mare/highland terrain (multi-octave fractal noise)
    dem = np.zeros((size, size), dtype=np.float32)
    scales = [size // 2, size // 4, size // 8, size // 16]
    amplitudes = [30.0, 15.0, 7.5, 3.0]

    for scale, amp in zip(scales, amplitudes):
        freq = 2.0 * np.pi / max(scale, 4)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        dem += amp * np.sin(freq * x + phase_x) * np.cos(freq * y + phase_y)

    # 2. Add realistic impact craters
    # Density determines number of craters
    if crater_density == "low":
        n_large, n_med, n_small = 2, 8, 20
    elif crater_density == "high":
        n_large, n_med, n_small = 6, 25, 70
    else:  # moderate
        n_large, n_med, n_small = 4, 16, 40

    crater_specs = []
    # Large craters: R ~ 35 - 70 px
    for _ in range(n_large):
        cx = rng.uniform(size * 0.15, size * 0.85)
        cy = rng.uniform(size * 0.15, size * 0.85)
        r = rng.uniform(35.0, 75.0)
        depth = r * 0.35
        rim = depth * 0.25
        crater_specs.append((cx, cy, r, depth, rim))

    # Medium craters: R ~ 15 - 35 px
    for _ in range(n_med):
        cx = rng.uniform(10, size - 10)
        cy = rng.uniform(10, size - 10)
        r = rng.uniform(15.0, 35.0)
        depth = r * 0.30
        rim = depth * 0.20
        crater_specs.append((cx, cy, r, depth, rim))

    # Small/micro craters: R ~ 5 - 15 px
    for _ in range(n_small):
        cx = rng.uniform(5, size - 5)
        cy = rng.uniform(5, size - 5)
        r = rng.uniform(5.0, 15.0)
        depth = r * 0.25
        rim = depth * 0.15
        crater_specs.append((cx, cy, r, depth, rim))

    # Imprint crater profiles onto DEM
    for cx, cy, r, depth, rim in crater_specs:
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        dist = np.sqrt(dist_sq)

        # Interior parabolic bowl
        bowl_mask = dist < r
        if np.any(bowl_mask):
            dem[bowl_mask] -= depth * (1.0 - (dist[bowl_mask] / r) ** 2)

        # Raised ejecta rim (Gaussian ring around radius r)
        rim_mask = (dist >= r * 0.8) & (dist < r * 2.0)
        if np.any(rim_mask):
            dem[rim_mask] += rim * np.exp(-((dist[rim_mask] - r) / (0.35 * r)) ** 2)

    # 3. Fine regolith micro-texture (high-frequency roughness)
    micro_texture = rng.normal(0.0, 0.8, (size, size)).astype(np.float32)
    dem += micro_texture

    return dem

def render_photometric_shading(
    dem: np.ndarray,
    sun_azimuth_deg: float = 45.0,
    sun_elevation_deg: float = 30.0,
    ambient: float = 0.08,
) -> np.ndarray:
    """
    Render 2D lunar surface radiance from elevation DEM using directional photometric shading.
    Computes surface normal vectors N = [-p, -q, 1] / ||N|| and solar dot product N . L.
    """
    # Compute surface slopes using central difference gradients
    grad_y, grad_x = np.gradient(dem)

    # Surface normal: N = [-dz/dx, -dz/dy, 1] normalized
    nx = -grad_x
    ny = -grad_y
    nz = np.ones_like(dem)

    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm

    # Solar illumination direction vector L = [Lx, Ly, Lz]
    sun_vec = sun_vector_from_angles(sun_azimuth_deg, sun_elevation_deg)
    if sun_vec is None:
        sun_vec = np.array([0.5, 0.5, 0.707], dtype=np.float32)

    lx, ly, lz = sun_vec[0], sun_vec[1], sun_vec[2]

    # Lambertian/diffuse cosine term: cos(theta_i) = N . L
    cos_i = nx * lx + ny * ly + nz * lz
    diffuse = np.maximum(0.0, cos_i)

    # Add ambient reflectance (Earthshine / inter-crater bounce)
    radiance = ambient + (1.0 - ambient) * diffuse

    # Convert to 8-bit grayscale [0, 255]
    radiance_u8 = (np.clip(radiance, 0.0, 1.0) * 255.0).astype(np.uint8)
    return radiance_u8
