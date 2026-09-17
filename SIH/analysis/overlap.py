"""
Tiered Overlap Estimation Module for Lunar Image Correspondence.
Implements three tiers of scientific honesty:
- Level 1: Metadata-assisted overlap (Geographic footprint intersection)
- Level 2: Image-based approximate overlap (Coarse cross-correlation)
- Level 3: Unknown / Insufficient information
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import cv2
from preprocessing.loader import LunarImage
from config import load_settings

OverlapConfidence = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]

@dataclass
class OverlapReport:
    """Standardized overlap estimation report."""
    overlap_pct: float | None
    level: str                  # "Level 1: Metadata-assisted" | "Level 2: Image-based approximate" | "Level 3: Unknown"
    method: str
    confidence: OverlapConfidence
    estimated_scale_ratio: float | None
    is_low_overlap: bool
    notes: str

def _estimate_level1_metadata_overlap(
    src: LunarImage,
    ref: LunarImage,
) -> OverlapReport | None:
    """
    Level 1: Compute true geographic footprint intersection if bounding coordinates exist.
    Expected metadata format in auxiliary: bbox = [min_lon, min_lat, max_lon, max_lat].
    """
    src_bbox = src.auxiliary.get("bbox")
    ref_bbox = ref.auxiliary.get("bbox")

    if not src_bbox or not ref_bbox or len(src_bbox) != 4 or len(ref_bbox) != 4:
        return None

    # [min_x, min_y, max_x, max_y]
    ix_min = max(src_bbox[0], ref_bbox[0])
    iy_min = max(src_bbox[1], ref_bbox[1])
    ix_max = min(src_bbox[2], ref_bbox[2])
    iy_max = min(src_bbox[3], ref_bbox[3])

    if ix_max <= ix_min or iy_max <= iy_min:
        return OverlapReport(
            overlap_pct=0.0,
            level="Level 1: Metadata-assisted",
            method="Geographic Footprint Intersection",
            confidence="HIGH",
            estimated_scale_ratio=None,
            is_low_overlap=True,
            notes="Footprints do not intersect geographically.",
        )

    inter_area = (ix_max - ix_min) * (iy_max - iy_min)
    src_area = (src_bbox[2] - src_bbox[0]) * (src_bbox[3] - src_bbox[1])
    ref_area = (ref_bbox[2] - ref_bbox[0]) * (ref_bbox[3] - ref_bbox[1])

    # Percentage relative to smaller image area
    min_area = min(src_area, ref_area)
    pct = round((inter_area / max(min_area, 1e-8)) * 100.0, 1)

    scale_ratio = None
    if isinstance(src.gsd, (int, float)) and isinstance(ref.gsd, (int, float)):
        scale_ratio = round(float(ref.gsd / src.gsd), 2)

    return OverlapReport(
        overlap_pct=pct,
        level="Level 1: Metadata-assisted",
        method="Geographic Footprint Intersection",
        confidence="HIGH",
        estimated_scale_ratio=scale_ratio,
        is_low_overlap=pct < 30.0,
        notes="Exact geographic overlap calculated from calibrated footprint metadata.",
    )

def _estimate_level2_image_overlap(
    src: LunarImage,
    ref: LunarImage,
    coarse_dim: int = 256,
) -> OverlapReport:
    """
    Level 2: Coarse 2D phase cross-correlation approximation.
    Strictly tagged as APPROXIMATE.
    """
    u8_src = src.to_uint8()
    u8_ref = ref.to_uint8()

    # Downsample to coarse resolution for robust translation search
    coarse_src = cv2.resize(u8_src, (coarse_dim, coarse_dim), interpolation=cv2.INTER_AREA).astype(np.float32)
    coarse_ref = cv2.resize(u8_ref, (coarse_dim, coarse_dim), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Windowing to reduce boundary spectral leakage in Fourier domain
    hann = cv2.createHanningWindow((coarse_dim, coarse_dim), cv2.CV_32F)
    coarse_src *= hann
    coarse_ref *= hann

    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(coarse_src, coarse_ref)
    except Exception:
        shift_x, shift_y, response = 0.0, 0.0, 0.0

    # If peak correlation response is too weak, fallback to Level 3
    if response < 0.04:
        return OverlapReport(
            overlap_pct=None,
            level="Level 3: Unknown / Insufficient Information",
            method="Coarse Phase Correlation Peak Failed",
            confidence="UNKNOWN",
            estimated_scale_ratio=None,
            is_low_overlap=True,
            notes="Correlation response below reliable detection threshold (response < 0.04).",
        )

    # Compute overlapping fraction
    overlap_w = max(0.0, coarse_dim - abs(shift_x))
    overlap_h = max(0.0, coarse_dim - abs(shift_y))
    overlap_area = overlap_w * overlap_h
    pct = round((overlap_area / (coarse_dim * coarse_dim)) * 100.0, 1)

    conf: OverlapConfidence = "MEDIUM" if response >= 0.12 else "LOW"

    scale_ratio = None
    if isinstance(src.gsd, (int, float)) and isinstance(ref.gsd, (int, float)):
        scale_ratio = round(float(ref.gsd / src.gsd), 2)

    return OverlapReport(
        overlap_pct=pct,
        level="Level 2: Image-based approximate",
        method="Coarse Phase Cross-Correlation",
        confidence=conf,
        estimated_scale_ratio=scale_ratio,
        is_low_overlap=pct < 35.0,
        notes=f"Approximate image-space overlap estimate (Correlation response={response:.3f}).",
    )

def estimate_overlap(
    source_lunar: LunarImage,
    reference_lunar: LunarImage,
    settings_override: dict | None = None,
) -> OverlapReport:
    """
    Tiered overlap estimation between source and reference lunar images.
    Tiers: Level 1 (Metadata) -> Level 2 (Image-based coarse) -> Level 3 (Unknown).
    """
    # Tier 1: Try metadata-assisted overlap
    meta_report = _estimate_level1_metadata_overlap(source_lunar, reference_lunar)
    if meta_report is not None:
        return meta_report

    # Tier 2: Try coarse phase correlation
    settings = settings_override or load_settings()
    coarse_size = settings.get("analysis", {}).get("overlap", {}).get("coarse_size", 256)
    return _estimate_level2_image_overlap(source_lunar, reference_lunar, coarse_dim=coarse_size)
