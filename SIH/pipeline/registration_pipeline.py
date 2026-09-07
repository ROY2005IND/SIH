"""
Central Registration Pipeline Orchestrator for MoonFlower AI.
Decouples scientific execution logic from UI presentation.
Coordinates Quality -> Overlap -> Preprocessing -> Matching -> Model Selection -> Spatial Balancing -> Sub-Pixel Refinement -> Metrics & Quality Scoring.
"""
from dataclasses import dataclass, field
import time
import numpy as np
from config import load_settings
from utils.logging import get_logger
from preprocessing.loader import LunarImage
from preprocessing.illumination import apply_preprocessing, PreprocessingMode
from analysis.image_quality import ImageQualityReport, analyze_image_quality
from analysis.overlap import OverlapReport, estimate_overlap
from matching.classical import ClassicalMatcher
from matching.result import MatchResult
from registration.model_selection import select_best_transformation_model
from registration.transformation import TransformationModel, calculate_residuals
from registration.subpixel import refine_subpixel_correspondences, SubPixelRefinementResult
from registration.ransac import estimate_robust_transformation
from registration.warping import (
    warp_image_to_reference,
    compute_difference_map,
    create_alpha_overlay,
)
from spatial.distribution import AdaptiveSpatialBalancer, SpatialBalancingResult
from evaluation.metrics import compute_registration_metrics, RegistrationMetrics
from evaluation.quality_score import calculate_quality_score, LunarRegistrationQualityScore

logger = get_logger("Pipeline")

@dataclass
class PipelineExecutionResult:
    """Encapsulates all intermediate and final scientific results from the registration pipeline."""
    success: bool
    match_result: MatchResult
    source_quality: ImageQualityReport
    reference_quality: ImageQualityReport
    overlap_report: OverlapReport
    spatial_result: SpatialBalancingResult | None = None
    subpixel_result: SubPixelRefinementResult | None = None
    metrics: RegistrationMetrics | None = None
    quality_score: LunarRegistrationQualityScore | None = None
    registered_image: np.ndarray | None = None
    difference_map: np.ndarray | None = None
    alpha_overlay: np.ndarray | None = None
    selected_model: str = "none"
    model_selection_reason: str = ""
    model_evaluations: dict = field(default_factory=dict)
    preprocessed_source: np.ndarray | None = None
    preprocessed_reference: np.ndarray | None = None
    execution_log: list[str] = field(default_factory=list)
    total_pipeline_time: float = 0.0

class RegistrationPipeline:
    """Central orchestrator coordinating the planetary image registration sequence."""

    def __init__(self, settings_override: dict | None = None):
        self.settings = settings_override or load_settings()

    def run(
        self,
        source: LunarImage,
        reference: LunarImage,
        preprocessing_method: PreprocessingMode = "clahe",
        matcher_method: str = "sift",
        ransac_threshold: float = 3.0,
        enable_spatial_balancing: bool = True,
        spatial_grid_size: int = 6,
        enable_subpixel: bool = True,
        candidate_models: list[TransformationModel] | None = None,
    ) -> PipelineExecutionResult:
        """Execute the complete scientific registration pipeline."""
        t_start = time.perf_counter()
        log_messages: list[str] = []

        def log_step(msg: str):
            logger.info(msg)
            log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        log_step(f"Initiating registration pipeline: Source({source.width}x{source.height}) -> Reference({reference.width}x{reference.height})")

        # 1. Image Quality Analysis
        log_step("Executing pre-matching radiometric & textural image quality analysis...")
        src_q = analyze_image_quality(source, self.settings)
        ref_q = analyze_image_quality(reference, self.settings)
        log_step(f"Source Quality: Status={src_q.readiness_status}, Texture={src_q.texture_level} ({src_q.texture_entropy} bits), Contrast={src_q.contrast_level}")
        log_step(f"Reference Quality: Status={ref_q.readiness_status}, Texture={ref_q.texture_level} ({ref_q.texture_entropy} bits), Contrast={ref_q.contrast_level}")

        # 2. Overlap Estimation
        log_step("Estimating spatial overlap between source and reference...")
        overlap_rep = estimate_overlap(source, reference, self.settings)
        pct_str = f"{overlap_rep.overlap_pct:.1f}%" if overlap_rep.overlap_pct is not None else "Unknown"
        log_step(f"Overlap Result: {pct_str} ({overlap_rep.level}, Confidence: {overlap_rep.confidence})")

        # 3. Preprocessing
        log_step(f"Applying '{preprocessing_method.upper()}' illumination-invariant preprocessing...")
        u8_src = source.to_uint8()
        u8_ref = reference.to_uint8()
        prep_cfg = self.settings.get("preprocessing", {})
        prep_src = apply_preprocessing(u8_src, method=preprocessing_method, params=prep_cfg)
        prep_ref = apply_preprocessing(u8_ref, method=preprocessing_method, params=prep_cfg)

        # 4. Feature Matching
        log_step(f"Executing '{matcher_method.upper()}' correspondence matching...")
        hybrid_diag = None
        if matcher_method.lower() == "hybrid":
            from matching.hybrid import AdaptiveHybridMatcher
            hybrid_engine = AdaptiveHybridMatcher(settings_override=self.settings)
            src_pts, ref_pts, confs, match_time, hybrid_diag = hybrid_engine.find_matches(
                prep_src, prep_ref, ransac_threshold=ransac_threshold
            )
            log_step(f"Hybrid Matcher: Fused {len(src_pts)} correspondences (SIFT: {hybrid_diag.sift_inliers}, Learned: {hybrid_diag.learned_inliers})")
        else:
            matcher = ClassicalMatcher(method=matcher_method, settings_override=self.settings)
            src_pts, ref_pts, confs, match_time = matcher.find_matches(prep_src, prep_ref)
            log_step(f"Detected {len(src_pts)} tentative correspondences in {match_time:.3f}s")

        if len(src_pts) < 4:
            log_step(f"CRITICAL: Insufficient correspondences ({len(src_pts)} < 4) to estimate transformation.")
            empty_res = MatchResult(
                method=matcher_method,
                source_points=src_pts,
                reference_points=ref_pts,
                confidences=confs,
                runtime_seconds=match_time,
            )
            return PipelineExecutionResult(
                success=False,
                match_result=empty_res,
                source_quality=src_q,
                reference_quality=ref_q,
                overlap_report=overlap_rep,
                preprocessed_source=prep_src,
                preprocessed_reference=prep_ref,
                execution_log=log_messages,
                total_pipeline_time=time.perf_counter() - t_start,
            )

        # 5. Robust Geometric Verification & Model Selection
        log_step("Performing geometric outlier rejection (USAC_MAGSAC) and model selection...")
        candidates = candidate_models or ["similarity", "affine", "homography"]
        try:
            best_model, M, inlier_mask, residuals, reason, evals = select_best_transformation_model(
                src_pts, ref_pts,
                candidate_models=candidates,
                ransac_threshold=ransac_threshold,
            )
            n_inliers = int(np.sum(inlier_mask))
            inlier_ratio = float(n_inliers / len(src_pts))
            log_step(f"Model Selection: {best_model.upper()} chosen. Reason: {reason}")
            log_step(f"Geometric Verification: {n_inliers}/{len(src_pts)} inliers ({inlier_ratio * 100:.1f}%)")
        except Exception as e:
            log_step(f"Geometric verification failed: {e}")
            fail_res = MatchResult(
                method=matcher_method,
                source_points=src_pts,
                reference_points=ref_pts,
                confidences=confs,
                runtime_seconds=match_time,
            )
            return PipelineExecutionResult(
                success=False,
                match_result=fail_res,
                source_quality=src_q,
                reference_quality=ref_q,
                overlap_report=overlap_rep,
                preprocessed_source=prep_src,
                preprocessed_reference=prep_ref,
                execution_log=log_messages,
                total_pipeline_time=time.perf_counter() - t_start,
            )

        # 6. Adaptive Spatial Match Balancer
        spatial_result: SpatialBalancingResult | None = None
        working_src = src_pts[inlier_mask]
        working_ref = ref_pts[inlier_mask]
        working_confs = confs[inlier_mask]
        working_res = residuals[inlier_mask] if residuals is not None else None

        if enable_spatial_balancing and len(working_src) >= 4:
            log_step(f"Applying Adaptive Spatial Match Balancer ({spatial_grid_size}x{spatial_grid_size} grid)...")
            balancer = AdaptiveSpatialBalancer(grid_size=spatial_grid_size, max_matches_per_cell=20)
            spatial_result = balancer.balance(
                working_src, working_ref, working_confs,
                (reference.height, reference.width),
                residuals=working_res,
            )
            log_step(
                f"Spatial Balancing: Uniformity improved from {spatial_result.raw_uniformity_score:.1f}% to "
                f"{spatial_result.balanced_uniformity_score:.1f}% ({len(spatial_result.balanced_source_points)} well-distributed points retained)"
            )
            working_src = spatial_result.balanced_source_points
            working_ref = spatial_result.balanced_reference_points

        # 7. Sub-Pixel Refinement
        subpixel_result: SubPixelRefinementResult | None = None
        if enable_subpixel and len(working_src) >= 4:
            log_step("Executing local patch NCC and 2D parabolic sub-pixel peak refinement...")
            subpixel_result = refine_subpixel_correspondences(
                u8_src, u8_ref,
                working_src, working_ref,
                patch_size=11, search_radius_px=2,
            )
            log_step(
                f"Sub-Pixel Refinement: {subpixel_result.n_converged}/{len(working_ref)} points refined. "
                f"Mean adjustment: {subpixel_result.mean_offset_px:.3f} px (Max: {subpixel_result.max_offset_px:.3f} px)"
            )
            # Re-estimate transformation using sub-pixel coordinates
            M_sub, mask_sub, res_sub = estimate_robust_transformation(
                working_src, subpixel_result.refined_points,
                model=best_model,
                ransac_threshold=ransac_threshold,
            )
            if M_sub is not None:
                M = M_sub
                working_ref = subpixel_result.refined_points

        # 8. Image Warping & Product Generation
        log_step("Warping source raster into reference coordinate frame...")
        warped_src = warp_image_to_reference(u8_src, M, (reference.height, reference.width))
        diff_map, mean_diff = compute_difference_map(warped_src, u8_ref)
        alpha_overlay = create_alpha_overlay(warped_src, u8_ref)
        log_step(f"Image warping completed. Mean absolute overlap difference: {mean_diff:.2f}")

        # 9. Compute Rigorous Metrics and Explainable Quality Score
        current_residuals = calculate_residuals(working_src, working_ref, M)
        full_inlier_mask = np.ones(len(working_src), dtype=bool)

        metrics = compute_registration_metrics(
            working_src, working_ref,
            full_inlier_mask, current_residuals,
            (reference.height, reference.width),
            runtime_seconds=time.perf_counter() - t_start,
            grid_size=spatial_grid_size,
        )
        # Preserve original tentative count
        metrics.tentative_matches = len(src_pts)
        metrics.inlier_ratio_pct = round((len(working_src) / max(len(src_pts), 1)) * 100.0, 2)

        quality_score = calculate_quality_score(metrics)
        log_step(f"Explainable Quality Score: {quality_score.total_score}/100 ({quality_score.rating_label})")

        # Assemble final MatchResult
        match_result = MatchResult(
            method=matcher_method,
            source_points=src_pts,
            reference_points=ref_pts,
            confidences=confs,
            inlier_mask=inlier_mask,
            transformation=M,
            transformation_type=best_model,
            residuals=residuals,
            runtime_seconds=match_time,
            diagnostics={
                "model_evaluations": evals,
                "reason": reason,
                "mean_overlap_diff": mean_diff,
                "spatial_uniformity": spatial_result.balanced_uniformity_score if spatial_result else None,
                "subpixel_converged": subpixel_result.n_converged if subpixel_result else None,
            },
        )

        total_time = time.perf_counter() - t_start
        log_step(f"Pipeline finished successfully in {total_time:.3f}s. Final Inlier RMSE: {metrics.rmse_px:.3f}px")

        return PipelineExecutionResult(
            success=True,
            match_result=match_result,
            source_quality=src_q,
            reference_quality=ref_q,
            overlap_report=overlap_rep,
            spatial_result=spatial_result,
            subpixel_result=subpixel_result,
            metrics=metrics,
            quality_score=quality_score,
            registered_image=warped_src,
            difference_map=diff_map,
            alpha_overlay=alpha_overlay,
            selected_model=best_model,
            model_selection_reason=reason,
            model_evaluations=evals,
            preprocessed_source=prep_src,
            preprocessed_reference=prep_ref,
            execution_log=log_messages,
            total_pipeline_time=total_time,
        )
