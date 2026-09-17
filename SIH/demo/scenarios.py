"""
Controlled Lunar-Like Synthetic Benchmark Scenarios.
Provides 7 mathematically verified test scenarios with model-specific Ground Truth transformations.
Labeled strictly as: Controlled Lunar-Like Synthetic Benchmark (Simplified Photometric Approximation).
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import cv2
from preprocessing.loader import LunarImage
from .generator import generate_lunar_terrain, render_photometric_shading

TransformationType = Literal["identity", "translation", "similarity", "affine", "homography"]

@dataclass
class ScenarioBenchmarkData:
    """Encapsulates a benchmark scenario pair with exact Ground Truth parameters."""
    scenario_id: str
    title: str
    category: str
    description: str
    source_image: LunarImage
    reference_image: LunarImage
    transformation_type: TransformationType
    ground_truth_matrix: np.ndarray   # Maps source coordinates [x, y] to reference coordinates [x, y]
    scale_factor: float
    rotation_deg: float
    sun_azimuth_source: float
    sun_azimuth_ref: float
    sun_elevation_source: float
    sun_elevation_ref: float
    sample_source_points: np.ndarray   # (K, 2) grid points in source frame
    sample_reference_points: np.ndarray# (K, 2) exact mapped points in reference frame

def _apply_affine_to_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 or 3x3 affine matrix M to (N, 2) points: P_ref = M * P_src."""
    if M.shape == (2, 3):
        return (pts @ M[:, :2].T) + M[:, 2]
    elif M.shape == (3, 3):
        homog = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        mapped = (homog @ M.T)
        return mapped[:, :2] / (mapped[:, 2:3] + 1e-8)
    raise ValueError(f"Invalid transformation matrix shape: {M.shape}")

def generate_scenario_pair(scenario_id: str, size: int = 512, seed: int = 42) -> ScenarioBenchmarkData:
    """
    Generate a controlled synthetic lunar benchmark image pair with known Ground Truth.
    """
    center = (size / 2.0, size / 2.0)
    grid_y, grid_x = np.mgrid[size * 0.25:size * 0.75:10j, size * 0.25:size * 0.75:10j]
    sample_pts_src = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    if scenario_id == "scenario_1_small_scale":
        title = "Scenario 1: Small Scale Variation (1.2x)"
        category = "Scale Invariance"
        description = "Controlled 1.2x zoom difference with identical solar illumination angle."
        dem = generate_lunar_terrain(size=size, seed=seed, crater_density="moderate")
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)

        # Scale 1.2x around center
        scale = 1.2
        # M_ref_to_src scales up, so M_src_to_ref = inv(M_ref_to_src)
        M_src_to_ref_2x3 = cv2.getRotationMatrix2D(center, 0.0, scale)
        M_ref_to_src_2x3 = cv2.getRotationMatrix2D(center, 0.0, 1.0 / scale)
        src_u8 = cv2.warpAffine(ref_u8, M_ref_to_src_2x3, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        M_gt = M_src_to_ref_2x3
        ttype: TransformationType = "similarity"
        rot_deg = 0.0
        az_src, az_ref = 45.0, 45.0
        el_src, el_ref = 30.0, 30.0

    elif scenario_id == "scenario_2_large_scale":
        title = "Scenario 2: Large Scale Variation (4.0x — OHRC vs TMC-2)"
        category = "Cross-Resolution"
        description = "Simulates high-resolution OHRC (~0.25 m) vs regional TMC-2 (1.0–5.0 m)."
        dem = generate_lunar_terrain(size=size, seed=seed + 1, crater_density="high")
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=60.0, sun_elevation_deg=25.0)

        scale = 4.0
        M_src_to_ref_2x3 = cv2.getRotationMatrix2D(center, 0.0, scale)
        M_ref_to_src_2x3 = cv2.getRotationMatrix2D(center, 0.0, 1.0 / scale)
        src_u8 = cv2.warpAffine(ref_u8, M_ref_to_src_2x3, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        M_gt = M_src_to_ref_2x3
        ttype = "similarity"
        rot_deg = 0.0
        az_src, az_ref = 60.0, 60.0
        el_src, el_ref = 25.0, 25.0

    elif scenario_id == "scenario_3_rotation_translation":
        title = "Scenario 3: Rotation & Translation"
        category = "Euclidean Pose"
        description = "In-plane orbit yaw rotation (25°) with ground track displacement (dx=20, dy=-15)."
        dem = generate_lunar_terrain(size=size, seed=seed + 2, crater_density="moderate")
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=35.0)

        rot_deg = 25.0
        dx, dy = 20.0, -15.0
        scale = 1.0
        M_rot = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy
        M_src_to_ref_2x3 = M_rot

        # Inverse mapping for rendering source
        M_3x3 = np.vstack([M_src_to_ref_2x3, [0, 0, 1]])
        M_inv_3x3 = np.linalg.inv(M_3x3)
        M_ref_to_src_2x3 = M_inv_3x3[:2, :]

        src_u8 = cv2.warpAffine(ref_u8, M_ref_to_src_2x3, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        M_gt = M_src_to_ref_2x3
        ttype = "similarity"
        az_src, az_ref = 45.0, 45.0
        el_src, el_ref = 35.0, 35.0

    elif scenario_id == "scenario_4_viewpoint_affine":
        title = "Scenario 4: Viewpoint / Oblique Affine Distortion"
        category = "Viewpoint Distortion"
        description = "Simulates off-nadir pitch/roll viewing geometry causing perspective foreshortening."
        dem = generate_lunar_terrain(size=size, seed=seed + 3, crater_density="moderate")
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)

        # Affine matrix with anisotropic scale and shear
        theta = np.radians(15.0)
        shear = 0.15
        sx, sy = 1.15, 0.88
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S = np.array([[sx, shear], [0.0, sy]])
        A = R @ S
        tx = center[0] - (A[0, 0] * center[0] + A[0, 1] * center[1]) + 10.0
        ty = center[1] - (A[1, 0] * center[0] + A[1, 1] * center[1]) - 8.0

        M_src_to_ref_2x3 = np.array([[A[0, 0], A[0, 1], tx], [A[1, 0], A[1, 1], ty]], dtype=np.float64)
        M_3x3 = np.vstack([M_src_to_ref_2x3, [0, 0, 1]])
        M_inv = np.linalg.inv(M_3x3)[:2, :]

        src_u8 = cv2.warpAffine(ref_u8, M_inv, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        M_gt = M_src_to_ref_2x3
        ttype = "affine"
        scale = float(np.sqrt(sx * sy))
        rot_deg = 15.0
        az_src, az_ref = 45.0, 45.0
        el_src, el_ref = 30.0, 30.0

    elif scenario_id == "scenario_5_illumination_shift":
        title = "Scenario 5: Sun-Angle & Shadow Direction Inversion"
        category = "Illumination Invariance"
        description = "Extreme shadow inversion: solar azimuth shifts 180° (45° vs 225°) across identical terrain."
        dem = generate_lunar_terrain(size=size, seed=seed + 4, crater_density="high")
        # Identical terrain, strictly different lighting angles!
        src_u8 = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=35.0)
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=225.0, sun_elevation_deg=18.0)

        # Spatial transformation is exact identity!
        M_gt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        ttype = "identity"
        scale = 1.0
        rot_deg = 0.0
        az_src, az_ref = 45.0, 225.0
        el_src, el_ref = 35.0, 18.0

    elif scenario_id == "scenario_6_low_texture_mare":
        title = "Scenario 6: Low-Texture Lunar Mare Plains"
        category = "Degraded Conditions"
        description = "Smooth lunar mare with minimal relief and subtle micro-craters testing feature sensitivity."
        dem = generate_lunar_terrain(size=size, seed=seed + 5, crater_density="low")
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=50.0, sun_elevation_deg=40.0)

        dx, dy = 18.0, -12.0
        M_src_to_ref_2x3 = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        M_ref_to_src_2x3 = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
        src_u8 = cv2.warpAffine(ref_u8, M_ref_to_src_2x3, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        M_gt = M_src_to_ref_2x3
        ttype = "translation"
        scale = 1.0
        rot_deg = 0.0
        az_src, az_ref = 50.0, 50.0
        el_src, el_ref = 40.0, 40.0

    else:  # scenario_7_combined
        scenario_id = "scenario_7_combined"
        title = "Scenario 7: Combined Multi-Challenge (Scale + Rotation + Lighting)"
        category = "Compound Stress Test"
        description = "Simultaneous scale (1.8x), rotation (30°), ground displacement, and solar azimuth shift (45° vs 135°)."
        dem = generate_lunar_terrain(size=size, seed=seed + 6, crater_density="high")
        src_base = render_photometric_shading(dem, sun_azimuth_deg=45.0, sun_elevation_deg=35.0)
        ref_u8 = render_photometric_shading(dem, sun_azimuth_deg=135.0, sun_elevation_deg=22.0)

        scale = 1.8
        rot_deg = 30.0
        M_src_to_ref_2x3 = cv2.getRotationMatrix2D(center, rot_deg, scale)
        M_src_to_ref_2x3[0, 2] += 12.0
        M_src_to_ref_2x3[1, 2] -= 10.0

        M_3x3 = np.vstack([M_src_to_ref_2x3, [0, 0, 1]])
        M_inv = np.linalg.inv(M_3x3)[:2, :]
        src_u8 = cv2.warpAffine(src_base, M_inv, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        M_gt = M_src_to_ref_2x3
        ttype = "similarity"
        az_src, az_ref = 45.0, 135.0
        el_src, el_ref = 35.0, 22.0

    # Map sample points to Ground Truth reference coordinates
    sample_pts_ref = _apply_affine_to_points(sample_pts_src, M_gt)

    src_lunar = LunarImage(
        image=src_u8,
        width=size,
        height=size,
        channels=1,
        mission="Synthetic Benchmark",
        instrument=f"Procedural Lunar Model ({title})",
        sensor="Simulated Lunar Camera (Source)",
        gsd=1.0,
        projection="Planar Orthographic (Synthetic GT)",
        acquisition_time="Controlled Simulation",
        sun_azimuth=az_src,
        sun_elevation=el_src,
        incidence_angle=round(90.0 - el_src, 1),
        emission_angle=0.0,
        phase_angle=round(90.0 - el_src, 1),
        wavelength="Panchromatic Visible (Simulated)",
        source_path=f"Synthetic::{scenario_id}::source",
    )

    ref_lunar = LunarImage(
        image=ref_u8,
        width=size,
        height=size,
        channels=1,
        mission="Synthetic Benchmark",
        instrument=f"Procedural Lunar Model ({title})",
        sensor="Simulated Lunar Camera (Reference)",
        gsd=round(1.0 * scale, 2),
        projection="Planar Orthographic (Synthetic GT)",
        acquisition_time="Controlled Simulation",
        sun_azimuth=az_ref,
        sun_elevation=el_ref,
        incidence_angle=round(90.0 - el_ref, 1),
        emission_angle=0.0,
        phase_angle=round(90.0 - el_ref, 1),
        wavelength="Panchromatic Visible (Simulated)",
        source_path=f"Synthetic::{scenario_id}::reference",
    )

    return ScenarioBenchmarkData(
        scenario_id=scenario_id,
        title=title,
        category=category,
        description=description,
        source_image=src_lunar,
        reference_image=ref_lunar,
        transformation_type=ttype,
        ground_truth_matrix=M_gt,
        scale_factor=scale,
        rotation_deg=rot_deg,
        sun_azimuth_source=az_src,
        sun_azimuth_ref=az_ref,
        sun_elevation_source=el_src,
        sun_elevation_ref=el_ref,
        sample_source_points=sample_pts_src,
        sample_reference_points=sample_pts_ref,
    )

SCENARIOS_CATALOG = [
    ("scenario_1_small_scale", "Scenario 1: Small Scale Variation (1.2x)"),
    ("scenario_2_large_scale", "Scenario 2: Large Scale Variation (4.0x)"),
    ("scenario_3_rotation_translation", "Scenario 3: Rotation & Translation"),
    ("scenario_4_viewpoint_affine", "Scenario 4: Viewpoint / Oblique Affine"),
    ("scenario_5_illumination_shift", "Scenario 5: Sun-Angle & Shadow Inversion"),
    ("scenario_6_low_texture_mare", "Scenario 6: Low-Texture Lunar Mare"),
    ("scenario_7_combined", "Scenario 7: Combined Multi-Challenge"),
]

def list_available_scenarios() -> list[tuple[str, str]]:
    """Return list of (scenario_id, title) tuples."""
    return SCENARIOS_CATALOG

def get_scenario_by_id(scenario_id: str, size: int = 512, seed: int = 42) -> ScenarioBenchmarkData:
    """Retrieve benchmark data for requested scenario ID."""
    return generate_scenario_pair(scenario_id, size=size, seed=seed)
