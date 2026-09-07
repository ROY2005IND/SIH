"""
Scientific Export and Reporting Engine for MoonFlower AI.
Exports full registration packages: CSV matches, JSON telemetry, GeoTIFF/PNG rasters,
and comprehensive Markdown/PDF scientific reports.
"""
from pathlib import Path
import json
import csv
import numpy as np
import cv2
from utils.paths import OUTPUTS_DIR
from preprocessing.loader import LunarImage
from .registration_pipeline import PipelineExecutionResult

def export_registration_package(
    result: PipelineExecutionResult,
    source: LunarImage,
    reference: LunarImage,
    export_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Export all imagery, data tables, and scientific reports to disk.
    Returns mapping of artifact names to file paths.
    """
    out_dir = export_dir or OUTPUTS_DIR / f"run_{int(result.total_pipeline_time * 1000)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    # 1. Export Imagery
    if result.registered_image is not None:
        reg_path = out_dir / "registered_source.png"
        cv2.imwrite(str(reg_path), result.registered_image)
        artifacts["registered_image"] = reg_path

    if result.difference_map is not None:
        diff_path = out_dir / "difference_map.png"
        cv2.imwrite(str(diff_path), result.difference_map)
        artifacts["difference_map"] = diff_path

    if result.alpha_overlay is not None:
        over_path = out_dir / "alignment_overlay.png"
        # Convert RGB to BGR for OpenCV write
        over_bgr = cv2.cvtColor(result.alpha_overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(over_path), over_bgr)
        artifacts["alpha_overlay"] = over_path

    # 2. Export Matches CSV
    m = result.match_result
    csv_path = out_dir / "correspondences.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "match_id",
            "source_x",
            "source_y",
            "reference_x",
            "reference_y",
            "confidence",
            "is_inlier",
            "residual_px",
            "refined_ref_x",
            "refined_ref_y",
        ])
        sub_pts = result.subpixel_result.refined_points if result.subpixel_result else m.reference_points
        for i in range(len(m.source_points)):
            is_inl = bool(m.inlier_mask[i]) if m.inlier_mask is not None else True
            res_val = float(m.residuals[i]) if m.residuals is not None else 0.0
            ref_x = float(sub_pts[i, 0]) if i < len(sub_pts) else float(m.reference_points[i, 0])
            ref_y = float(sub_pts[i, 1]) if i < len(sub_pts) else float(m.reference_points[i, 1])

            writer.writerow([
                i,
                round(float(m.source_points[i, 0]), 3),
                round(float(m.source_points[i, 1]), 3),
                round(float(m.reference_points[i, 0]), 3),
                round(float(m.reference_points[i, 1]), 3),
                round(float(m.confidences[i]), 3),
                is_inl,
                round(res_val, 3),
                round(ref_x, 3),
                round(ref_y, 3),
            ])
    artifacts["matches_csv"] = csv_path

    # 3. Export Transformation Matrix JSON
    trans_path = out_dir / "transformation.json"
    trans_dict = {
        "transformation_model": result.selected_model,
        "selection_reason": result.model_selection_reason,
        "matrix": result.match_result.transformation.tolist() if result.match_result.transformation is not None else None,
        "evaluations": result.model_evaluations,
    }
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(trans_dict, f, indent=2)
    artifacts["transformation_json"] = trans_path

    # 4. Export Scientific Metrics JSON
    metrics_path = out_dir / "metrics.json"
    if result.metrics:
        met = result.metrics
        met_dict = {
            "tentative_matches": met.tentative_matches,
            "inlier_count": met.inlier_count,
            "inlier_ratio_pct": met.inlier_ratio_pct,
            "rmse_px": met.rmse_px,
            "median_residual_px": met.median_residual_px,
            "p90_residual_px": met.p90_residual_px,
            "max_inlier_error_px": met.max_inlier_error_px,
            "spatial_coverage_pct": met.spatial_coverage_pct,
            "spatial_uniformity_score": met.spatial_uniformity_score,
            "grid_occupancy_pct": met.grid_occupancy_pct,
            "runtime_seconds": met.runtime_seconds,
            "quality_score": result.quality_score.total_score if result.quality_score else None,
            "quality_rating": result.quality_score.rating_label if result.quality_score else None,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(met_dict, f, indent=2)
        artifacts["metrics_json"] = metrics_path

    # 5. Generate Professional Scientific Report Markdown
    report_path = out_dir / "LUNAR_REGISTRATION_REPORT.md"
    rep_text = f"""# MoonFlower AI — Planetary Scientific Registration Report
**Problem Statement:** ISRO SIH #26166  
**Registration Status:** {'SUCCESSFUL' if result.success else 'FAILED'}  
**Total Pipeline Execution Time:** {result.total_pipeline_time:.3f} seconds  

---

## 1. Input Datasets & Planetary Metadata

| Metadata Parameter | Source Sensor | Reference Sensor |
| :--- | :--- | :--- |
| **Mission** | {source.mission} | {reference.mission} |
| **Instrument** | {source.instrument} | {reference.instrument} |
| **Sensor Tag** | {source.sensor} | {reference.sensor} |
| **Dimensions** | {source.width} x {source.height} | {reference.width} x {reference.height} |
| **Ground Sample Distance (GSD)** | {source.gsd} m/px | {reference.gsd} m/px |
| **Solar Azimuth / Elevation** | {source.sun_azimuth}° / {source.sun_elevation}° | {reference.sun_azimuth}° / {reference.sun_elevation}° |
| **Projection** | {source.projection} | {reference.projection} |

---

## 2. Pre-Matching Analysis & Overlap

- **Source Image Quality:** Status: `{result.source_quality.readiness_status}`, Texture: `{result.source_quality.texture_level}` ({result.source_quality.texture_entropy} bits), Contrast: `{result.source_quality.contrast_level}`
- **Reference Image Quality:** Status: `{result.reference_quality.readiness_status}`, Texture: `{result.reference_quality.texture_level}` ({result.reference_quality.texture_entropy} bits), Contrast: `{result.reference_quality.contrast_level}`
- **Spatial Overlap Estimation:** {result.overlap_report.overlap_pct or 'N/A'}% ({result.overlap_report.level}, Confidence: `{result.overlap_report.confidence}`)

---

## 3. Registration Performance Metrics

- **Tentative Correspondences:** {result.metrics.tentative_matches if result.metrics else 0}
- **Geometrically Verified Inliers:** {result.metrics.inlier_count if result.metrics else 0}
- **Inlier Ratio:** {result.metrics.inlier_ratio_pct if result.metrics else 0.0}%
- **Transfer RMSE:** **{result.metrics.rmse_px if result.metrics else 'N/A'} px**
- **Median Residual Error:** {result.metrics.median_residual_px if result.metrics else 'N/A'} px
- **Spatial Coverage:** {result.metrics.spatial_coverage_pct if result.metrics else 0.0}%
- **Spatial Uniformity Score:** {result.metrics.spatial_uniformity_score if result.metrics else 0.0} / 100
- **Internal Quality Score:** **{result.quality_score.total_score if result.quality_score else 0} / 100 ({result.quality_score.rating_label if result.quality_score else 'N/A'})**
  *(Note: Internal experimental score — not an official ISRO metric)*

---

## 4. Geometric Model Selection

- **Selected Transformation:** `{result.selected_model.upper()}`
- **Occam's Razor Justification:** {result.model_selection_reason}

---

## 5. Sub-Pixel Refinement & Spatial Balancing

- **Spatial Match Balancing:** {'Enabled' if result.spatial_result else 'Bypassed'}
  - Raw Uniformity: {result.spatial_result.raw_uniformity_score if result.spatial_result else 'N/A'}% -> Balanced Uniformity: {result.spatial_result.balanced_uniformity_score if result.spatial_result else 'N/A'}%
- **Sub-Pixel Peak Optimization:** {'Enabled' if result.subpixel_result else 'Bypassed'}
  - Refined Points: {result.subpixel_result.n_converged if result.subpixel_result else 0}
  - Measured Mean Adjustment: {result.subpixel_result.mean_offset_px if result.subpixel_result else 0.0} px
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rep_text)
    artifacts["report_md"] = report_path

    return artifacts
