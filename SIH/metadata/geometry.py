"""Planetary solar and camera geometry calculations for lunar image correspondence."""
import numpy as np

def sun_vector_from_angles(azimuth_deg: float | str, elevation_deg: float | str) -> np.ndarray | None:
    """
    Calculate 3D unit illumination vector L = [Lx, Ly, Lz] from solar azimuth and elevation.
    Frame: East-North-Up (ENU).
    Azimuth: Measured clockwise from North (0° = North, 90° = East).
    Elevation: Measured from local lunar horizontal plane (0° = Horizon, 90° = Zenith).
    """
    try:
        az = float(azimuth_deg)
        el = float(elevation_deg)
    except (ValueError, TypeError):
        return None

    az_rad = np.radians(az)
    el_rad = np.radians(el)

    lx = np.cos(el_rad) * np.sin(az_rad)  # East
    ly = np.cos(el_rad) * np.cos(az_rad)  # North
    lz = np.sin(el_rad)                   # Up

    vec = np.array([lx, ly, lz], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)

def calculate_gsd_scale_ratio(source_gsd: float | str, ref_gsd: float | str) -> float | None:
    """
    Calculate theoretical geometric scale ratio S = ref_gsd / source_gsd.
    For example, OHRC (0.25 m/px) vs TMC-2 (5.0 m/px) yields scale ratio 20.0x.
    """
    try:
        s_gsd = float(source_gsd)
        r_gsd = float(ref_gsd)
        if s_gsd <= 0 or r_gsd <= 0:
            return None
        return float(r_gsd / s_gsd)
    except (ValueError, TypeError):
        return None

def solar_phase_angle(incidence_deg: float | str, emission_deg: float | str, rel_azimuth_deg: float | str) -> float | None:
    """
    Calculate solar phase angle (alpha) from incidence (i), emission (e), and relative azimuth (phi).
    cos(alpha) = cos(i)*cos(e) + sin(i)*sin(e)*cos(phi).
    """
    try:
        i = np.radians(float(incidence_deg))
        e = np.radians(float(emission_deg))
        phi = np.radians(float(rel_azimuth_deg))
        cos_alpha = np.cos(i) * np.cos(e) + np.sin(i) * np.sin(e) * np.cos(phi)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_alpha)))
    except (ValueError, TypeError):
        return None
