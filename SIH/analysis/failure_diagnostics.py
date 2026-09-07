"""
Automated Failure Analysis & Scientific Diagnostic Engine for Planetary Registration.
Evaluates root causes of poor correspondence or geometric failure and provides rule-based recommendations.
"""
from dataclasses import dataclass, field
from typing import Literal
from .image_quality import ImageQualityReport
from .overlap import OverlapReport
from matching.result import MatchResult
from evaluation.metrics import RegistrationMetrics

FailureCategory = Literal[
    "LOW_TEXTURE",
    "LARGE_SCALE_DIFFERENCE",
    "LOW_SPATIAL_OVERLAP",
    "SEVERE_ILLUMINATION_INVERSION",
    "HIGH_VIEWPOINT_DISTORTION",
    "POOR_SPATIAL_COVERAGE",
    "INSUFFICIENT_MATCHES",
    "EXCESSIVE_GEOMETRIC_STRAIN",
    "NOMINAL",
]

@dataclass
class DiagnosticFinding:
    category: FailureCategory
    severity: Literal["CRITICAL", "WARNING", "INFO"]
    observation: str
    recommendation: str

@dataclass
class FailureAnalysisReport:
    is_success: bool
    primary_issue: str
    findings: list[DiagnosticFinding]
    recommended_parameters: dict[str, str]

def diagnose_registration_run(
    source_quality: ImageQualityReport,
    reference_quality: ImageQualityReport,
    overlap_report: OverlapReport,
    match_result: MatchResult,
    metrics: RegistrationMetrics | None = None,
    sun_azimuth_delta_deg: float | None = None,
) -> FailureAnalysisReport:
    """
    Perform deep rule-based scientific diagnostics on a registration attempt.
    Analyzes radiometric contrast, scale ratios, sun angles, inlier ratios, and coverage.
    """
    findings: list[DiagnosticFinding] = []
    rec_params: dict[str, str] = {}

    # 1. Texture & Contrast Check
    if source_quality.texture_level == "LOW" or reference_quality.texture_level == "LOW":
        findings.append(DiagnosticFinding(
            category="LOW_TEXTURE",
            severity="CRITICAL" if match_result.n_inliers < 10 else "WARNING",
            observation=f"Low textural entropy detected (Source: {source_quality.texture_entropy} bits, Ref: {reference_quality.texture_entropy} bits). Typical of smooth lunar maria.",
            recommendation="Switch to 'Adaptive Hybrid' correspondence engine or lower feature contrast threshold to boost keypoint density across subtle relief.",
        ))
        rec_params["matcher"] = "hybrid"

    if source_quality.contrast_level == "LOW" or reference_quality.contrast_level == "LOW":
        findings.append(DiagnosticFinding(
            category="LOW_TEXTURE",
            severity="WARNING",
            observation=f"Low radiometric contrast detected (Source std: {source_quality.contrast_std}, Ref std: {reference_quality.contrast_std}).",
            recommendation="Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization) preprocessing to enhance local micro-craters.",
        ))
        rec_params["preprocessing"] = "clahe"

    # 2. Scale Difference Check
    if overlap_report.estimated_scale_ratio and overlap_report.estimated_scale_ratio > 2.5:
        findings.append(DiagnosticFinding(
            category="LARGE_SCALE_DIFFERENCE",
            severity="WARNING",
            observation=f"Cross-sensor resolution gap of {overlap_report.estimated_scale_ratio:.1f}x detected (e.g. OHRC vs TMC-2).",
            recommendation="Enable Multi-Scale Gaussian Pyramid search (coarse-to-fine) to establish correspondence across octave scales.",
        ))
        rec_params["pyramid"] = "enabled"

    # 3. Solar Illumination Inversion Check
    if sun_azimuth_delta_deg is not None and abs(sun_azimuth_delta_deg) > 120.0:
        findings.append(DiagnosticFinding(
            category="SEVERE_ILLUMINATION_INVERSION",
            severity="WARNING",
            observation=f"Solar azimuth angle shifts by {abs(sun_azimuth_delta_deg):.0f}° causing crater shadow inversion.",
            recommendation="Select 'Sobel Gradient Magnitude' or 'Phase Congruency' preprocessing to achieve illumination invariance on topographic rim boundaries.",
        ))
        rec_params["preprocessing"] = "gradient"

    # 4. Overlap Check
    if overlap_report.overlap_pct is not None and overlap_report.overlap_pct < 35.0:
        findings.append(DiagnosticFinding(
            category="LOW_SPATIAL_OVERLAP",
            severity="CRITICAL" if match_result.n_inliers < 8 else "WARNING",
            observation=f"Estimated spatial overlap is marginal ({overlap_report.overlap_pct:.1f}%).",
            recommendation="Verify orbit ground tracks or provide higher-overlap reference imagery.",
        ))

    # 5. Spatial Coverage Check
    if metrics and metrics.spatial_coverage_pct < 25.0 and metrics.inlier_count >= 4:
        findings.append(DiagnosticFinding(
            category="POOR_SPATIAL_COVERAGE",
            severity="WARNING",
            observation=f"Inliers are clustered within only {metrics.spatial_coverage_pct:.1f}% of the total image area.",
            recommendation="Enable 8x8 Adaptive Spatial Match Balancer with intra-cell spacing constraints to prevent crater rim clustering.",
        ))
        rec_params["spatial_grid"] = "8"

    # 6. Residual Strain Check
    if metrics and metrics.rmse_px and metrics.rmse_px > 3.0:
        findings.append(DiagnosticFinding(
            category="EXCESSIVE_GEOMETRIC_STRAIN",
            severity="WARNING",
            observation=f"High transfer RMSE ({metrics.rmse_px:.2f} px). Surface geometry exhibits non-planar parallax or high-order distortion.",
            recommendation="Upgrade transformation model to 'Homography' or inspect off-nadir emission angle metadata.",
        ))
        rec_params["model"] = "homography"

    # Overall summary
    if not findings:
        primary = "Nominal Lunar Imaging Conditions: Radiometric and spatial criteria satisfied."
        findings.append(DiagnosticFinding(
            category="NOMINAL",
            severity="INFO",
            observation="No critical operational anomalies detected.",
            recommendation="Standard SIFT or Hybrid registration pipeline optimal.",
        ))
    else:
        critical_findings = [f for f in findings if f.severity == "CRITICAL"]
        if critical_findings:
            primary = f"Primary Operational Bottleneck: {critical_findings[0].category} ({critical_findings[0].observation})"
        else:
            primary = f"Operational Consideration: {findings[0].category} ({findings[0].observation})"

    is_success = match_result.n_inliers >= 4 and (metrics.rmse_px is None or metrics.rmse_px < 4.0 if metrics else True)

    return FailureAnalysisReport(
        is_success=is_success,
        primary_issue=primary,
        findings=findings,
        recommended_parameters=rec_params,
    )
