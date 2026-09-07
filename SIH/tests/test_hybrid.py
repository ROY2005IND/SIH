"""Test suite for Phase 10/11 Adaptive Hybrid Matcher, Failure Diagnostics, and Export Package."""
import numpy as np
import pytest
from matching.hybrid import AdaptiveHybridMatcher
from analysis.failure_diagnostics import diagnose_registration_run
from analysis.image_quality import analyze_image_quality
from analysis.overlap import estimate_overlap
from pipeline.export import export_registration_package
from pipeline.registration_pipeline import RegistrationPipeline
from demo.scenarios import get_scenario_by_id

def test_adaptive_hybrid_matcher():
    """Verify hybrid matcher combines classical and learned candidates with evidence analysis."""
    scenario = get_scenario_by_id("scenario_1_small_scale", size=256, seed=42)
    hybrid = AdaptiveHybridMatcher()

    src_pts, ref_pts, confs, runtime, diag = hybrid.find_matches(
        scenario.source_image.to_uint8(),
        scenario.reference_image.to_uint8(),
    )

    assert len(src_pts) > 10
    assert len(src_pts) == len(ref_pts) == len(confs)
    assert diag.sift_inliers > 0
    assert diag.fused_total >= diag.sift_inliers
    assert "Adaptive Hybrid" in diag.decision_rationale

def test_failure_diagnostics_engine():
    """Verify diagnostic engine flags illumination inversion and low texture."""
    scenario5 = get_scenario_by_id("scenario_5_illumination_shift", size=256, seed=42)
    src_q = analyze_image_quality(scenario5.source_image)
    ref_q = analyze_image_quality(scenario5.reference_image)
    ov = estimate_overlap(scenario5.source_image, scenario5.reference_image)

    from matching.result import MatchResult
    mock_res = MatchResult(
        method="sift",
        source_points=np.array([[10, 10]], dtype=np.float32),
        reference_points=np.array([[10, 10]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
    )

    diag_rep = diagnose_registration_run(
        src_q, ref_q, ov, mock_res,
        sun_azimuth_delta_deg=180.0,
    )

    categories = [f.category for f in diag_rep.findings]
    assert "SEVERE_ILLUMINATION_INVERSION" in categories
    assert "gradient" in diag_rep.recommended_parameters.get("preprocessing", "")

def test_export_package(tmp_path):
    """Verify export package generates CSV, JSON, and Markdown report."""
    scenario = get_scenario_by_id("scenario_1_small_scale", size=128, seed=42)
    pipeline = RegistrationPipeline()
    res = pipeline.run(scenario.source_image, scenario.reference_image)

    artifacts = export_registration_package(
        res, scenario.source_image, scenario.reference_image, export_dir=tmp_path
    )

    assert "matches_csv" in artifacts and artifacts["matches_csv"].exists()
    assert "metrics_json" in artifacts and artifacts["metrics_json"].exists()
    assert "transformation_json" in artifacts and artifacts["transformation_json"].exists()
    assert "report_md" in artifacts and artifacts["report_md"].exists()
