"""
Central Registration Pipeline Orchestrator for LUNARMATCH AI.
Decouples scientific execution logic from UI presentation.
Coordinates Quality -> Overlap -> Preprocessing -> Matching -> Model Selection -> Spatial Balancing -> Sub-Pixel Refinement -> Metrics & Quality Scoring.

Stage-level point-count tracking:
    tentative_matches -> geometric_inliers -> spatially_balanced -> subpixel_verified -> final_inliers
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
    # Stage-level point counts for transparent reporting
    stage_counts: dict = field(default_factory=dict)
    # Stage-level execution timings in seconds
    stage_timings: dict[str, float] = field(default_factory=dict)
    # Scientific operating state: VERIFIED_GEOSPATIAL, IMAGE_ONLY_EXPLORATORY, or REJECTED
    operating_state: str = "VERIFIED_GEOSPATIAL"
    # Explicit rejection reason when registration is not accepted
    rejection_reason: str = ""

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
        stage_timings: dict[str, float] = {}

        # Stage-level point counts for transparent reporting
        stage_counts: dict[str, int] = {
            "tentative_matches": 0,
            "geometric_inliers": 0,
            "spatially_balanced": 0,
            "subpixel_candidates": 0,
            "subpixel_verified": 0,
            "final_inliers": 0,
            "rejected_matches": 0,
        }

        def log_step(msg: str):
            logger.info(msg)
            log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        log_step(f"Initiating registration pipeline: Source({source.width}x{source.height}) -> Reference({reference.width}x{reference.height})")

        # 1. Image Quality Analysis
        t_stage = time.perf_counter()
        log_step("Executing pre-matching radiometric & textural image quality analysis...")
        src_q = analyze_image_quality(source, self.settings)
        ref_q = analyze_image_quality(reference, self.settings)
        stage_timings["quality_analysis_seconds"] = round(time.perf_counter() - t_stage, 4)
        log_step(f"Source Quality: Status={src_q.readiness_status}, Texture={src_q.texture_level} ({src_q.texture_entropy} bits), Contrast={src_q.contrast_level}")
        log_step(f"Reference Quality: Status={ref_q.readiness_status}, Texture={ref_q.texture_level} ({ref_q.texture_entropy} bits), Contrast={ref_q.contrast_level}")

        # 2. Overlap Estimation
        t_stage = time.perf_counter()
        log_step("Estimating spatial overlap between source and reference...")
        overlap_rep = estimate_overlap(source, reference, self.settings)
        stage_timings["overlap_estimation_seconds"] = round(time.perf_counter() - t_stage, 4)
        pct_str = f"{overlap_rep.overlap_pct:.1f}%" if overlap_rep.overlap_pct is not None else "Unknown"
        log_step(f"Overlap Result: {pct_str} ({overlap_rep.level}, Confidence: {overlap_rep.confidence})")

        # 3. Preprocessing
        t_stage = time.perf_counter()
        log_step(f"Applying '{preprocessing_method.upper()}' illumination-invariant preprocessing...")
        u8_src = source.to_uint8()
        u8_ref = reference.to_uint8()
        prep_cfg = self.settings.get("preprocessing", {})
        prep_src = apply_preprocessing(u8_src, method=preprocessing_method, params=prep_cfg)
        prep_ref = apply_preprocessing(u8_ref, method=preprocessing_method, params=prep_cfg)
        stage_timings["preprocessing_seconds"] = round(time.perf_counter() - t_stage, 4)

        # 4. Feature Matching
        t_stage = time.perf_counter()
        # Detect pixel scale ratio — large ISRO strips (e.g. TMC-2 vs OHRC) often have >6x ratio
        pixel_scale_ratio = max(
            reference.height / max(source.height, 1),
            reference.width  / max(source.width,  1),
        )
        scale_aware_diag: dict = {}
        effective_ransac = ransac_threshold

        # Scale-Aware coarse-to-fine pre-pass for extreme scale disparities
        scale_aware_src_pts: np.ndarray = np.empty((0, 2), dtype=np.float32)
        scale_aware_ref_pts: np.ndarray = np.empty((0, 2), dtype=np.float32)
        scale_aware_confs:   np.ndarray = np.empty((0,),   dtype=np.float32)
        scale_aware_time: float = 0.0

        if pixel_scale_ratio >= 3.0:
            log_step(
                f"Large pixel scale ratio detected ({pixel_scale_ratio:.1f}x). "
                f"Activating Scale-Aware Coarse-to-Fine Matcher..."
            )
            from matching.scale_aware import ScaleAwareMatcher
            sa_matcher = ScaleAwareMatcher()
            scale_aware_src_pts, scale_aware_ref_pts, scale_aware_confs, scale_aware_time, scale_aware_diag = \
                sa_matcher.find_matches(prep_src, prep_ref, ransac_threshold=ransac_threshold)
            log_step(
                f"Scale-Aware Matcher: {len(scale_aware_src_pts)} correspondences found in {scale_aware_time:.2f}s "
                f"(NCC conf={scale_aware_diag.get('ncc_confidence', 'n/a')}, ROI={scale_aware_diag.get('roi_native', 'full')})"
            )
            # Relax RANSAC threshold proportionally for large pixel scales
            effective_ransac = max(ransac_threshold, min(ransac_threshold * (pixel_scale_ratio / 3.0), ransac_threshold * 5.0))
            if effective_ransac != ransac_threshold:
                log_step(f"RANSAC threshold relaxed to {effective_ransac:.1f}px for large-scale image pair.")

        log_step(f"Executing '{matcher_method.upper()}' correspondence matching...")
        hybrid_diag = None
        if matcher_method.lower() == "hybrid":
            from matching.hybrid import AdaptiveHybridMatcher
            hybrid_engine = AdaptiveHybridMatcher(settings_override=self.settings)
            src_pts, ref_pts, confs, match_time, hybrid_diag = hybrid_engine.find_matches(
                prep_src, prep_ref, ransac_threshold=effective_ransac
            )
            log_step(f"Hybrid Matcher: Fused {len(src_pts)} correspondences (SIFT: {hybrid_diag.sift_inliers}, Learned: {hybrid_diag.learned_inliers})")
        else:
            matcher = ClassicalMatcher(method=matcher_method, settings_override=self.settings)
            src_pts, ref_pts, confs, match_time = matcher.find_matches(prep_src, prep_ref)
            log_step(f"Detected {len(src_pts)} tentative correspondences in {match_time:.3f}s")

        # Fuse with scale-aware matches if they outperform or supplement
        if len(scale_aware_src_pts) > 0:
            if len(src_pts) < 10 or len(scale_aware_src_pts) > len(src_pts):
                log_step(
                    f"Using Scale-Aware matches ({len(scale_aware_src_pts)}) "
                    f"over classical ({len(src_pts)}) — better coverage detected."
                )
                src_pts, ref_pts, confs, match_time = (
                    scale_aware_src_pts, scale_aware_ref_pts, scale_aware_confs, scale_aware_time
                )
            elif len(scale_aware_src_pts) >= 4:
                log_step(f"Fusing {len(scale_aware_src_pts)} scale-aware matches into classical pool.")
                src_pts   = np.vstack([src_pts,   scale_aware_src_pts])
                ref_pts   = np.vstack([ref_pts,   scale_aware_ref_pts])
                confs     = np.concatenate([confs, scale_aware_confs * 0.9])

        stage_timings["feature_matching_seconds"] = round(time.perf_counter() - t_stage, 4)

        # Record tentative match count
        stage_counts["tentative_matches"] = len(src_pts)

        if len(src_pts) < 4:
            log_step(f"CRITICAL: Insufficient correspondences ({len(src_pts)} < 4) to estimate transformation.")
            stage_timings["total_seconds"] = round(time.perf_counter() - t_start, 4)
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
                stage_counts=stage_counts,
                stage_timings=stage_timings,
                operating_state="REJECTED",
                rejection_reason=f"Insufficient correspondences detected: only {len(src_pts)} tentative match(es) found (minimum 4 required).",
            )

        # 5. Robust Geometric Verification & Model Selection
        t_stage = time.perf_counter()
        candidates = candidate_models or ["similarity", "affine", "homography"]
        log_step("Performing geometric outlier rejection and model selection...")
        try:
            best_model, M, inlier_mask, residuals, reason, evals = select_best_transformation_model(
                src_pts, ref_pts,
                candidate_models=candidates,
                ransac_threshold=effective_ransac,
            )
            n_geometric_inliers = int(np.sum(inlier_mask))
            geometric_inlier_ratio = float(n_geometric_inliers / len(src_pts))
            stage_counts["geometric_inliers"] = min(stage_counts["tentative_matches"], n_geometric_inliers)

            # Report the actual RANSAC method used per model type
            ransac_method_label = "USAC_MAGSAC" if best_model == "homography" else "OpenCV_RANSAC"
            log_step(f"Model Selection: {best_model.upper()} chosen via {ransac_method_label}. Reason: {reason}")
            log_step(f"Geometric Verification: {n_geometric_inliers}/{len(src_pts)} inliers ({geometric_inlier_ratio * 100:.1f}%)")

            if n_geometric_inliers < 4 or (geometric_inlier_ratio < 0.05 and n_geometric_inliers < 6):
                log_step(
                    f"CRITICAL: Geometric verification rejected: insufficient consensus "
                    f"({n_geometric_inliers} inliers, {geometric_inlier_ratio * 100:.1f}% ratio). "
                    "Images appear unrelated, non-overlapping, or lack verifiable correspondences."
                )
                stage_timings["geometric_verification_seconds"] = round(time.perf_counter() - t_stage, 4)
                stage_timings["total_seconds"] = round(time.perf_counter() - t_start, 4)
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
                    stage_counts=stage_counts,
                    stage_timings=stage_timings,
                    operating_state="REJECTED",
                    rejection_reason=f"Geometric verification rejected: insufficient consensus ({n_geometric_inliers} inliers, {geometric_inlier_ratio * 100:.1f}% ratio).",
                )
        except Exception as e:
            log_step(f"Geometric verification failed: {e}")
            stage_timings["geometric_verification_seconds"] = round(time.perf_counter() - t_stage, 4)
            stage_timings["total_seconds"] = round(time.perf_counter() - t_start, 4)
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
                stage_counts=stage_counts,
                stage_timings=stage_timings,
                operating_state="REJECTED",
                rejection_reason=f"Geometric model estimation failed: {e}",
            )
        stage_timings["geometric_verification_seconds"] = round(time.perf_counter() - t_stage, 4)

        # 6. Adaptive Spatial Match Balancer
        t_stage = time.perf_counter()
        spatial_result: SpatialBalancingResult | None = None
        working_src = src_pts[inlier_mask]
        working_ref = ref_pts[inlier_mask]
        working_confs = confs[inlier_mask]
        working_res = residuals[inlier_mask] if residuals is not None else None

        if enable_spatial_balancing and len(working_src) >= 4:
            log_step(f"Applying Adaptive Spatial Match Balancer ({spatial_grid_size}x{spatial_grid_size} grid)...")
            balancer = AdaptiveSpatialBalancer(grid_size=spatial_grid_size, max_matches_per_cell=20)
            inlier_residuals = calculate_residuals(working_src, working_ref, M) if working_res is None else working_res
            spatial_result = balancer.balance(
                working_src, working_ref, working_confs,
                (reference.height, reference.width),
                residuals=inlier_residuals,
            )
            log_step(
                f"Spatial Balancing: Uniformity {spatial_result.raw_uniformity_score:.1f}% -> "
                f"{spatial_result.balanced_uniformity_score:.1f}% "
                f"({len(spatial_result.balanced_source_points)} well-distributed points retained)"
            )
            working_src = spatial_result.balanced_source_points
            working_ref = spatial_result.balanced_reference_points

        stage_counts["spatially_balanced"] = min(stage_counts["geometric_inliers"], len(working_src))
        stage_timings["spatial_balancing_seconds"] = round(time.perf_counter() - t_stage, 4)

        # 7. Sub-Pixel Refinement
        t_stage = time.perf_counter()
        subpixel_result: SubPixelRefinementResult | None = None
        final_inlier_mask: np.ndarray | None = None

        if enable_subpixel and len(working_src) >= 4:
            log_step("Executing local patch NCC and 2D parabolic sub-pixel peak refinement...")
            stage_counts["subpixel_candidates"] = len(working_src)

            subpixel_result = refine_subpixel_correspondences(
                u8_src, u8_ref,
                working_src, working_ref,
                patch_size=11, search_radius_px=2,
            )
            log_step(
                f"Sub-Pixel Refinement: {subpixel_result.n_converged}/{len(working_ref)} points refined. "
                f"Mean adjustment: {subpixel_result.mean_offset_px:.3f} px (Max: {subpixel_result.max_offset_px:.3f} px)"
            )
            stage_counts["subpixel_verified"] = subpixel_result.n_converged

            M_sub, mask_sub, res_sub = estimate_robust_transformation(
                working_src, subpixel_result.refined_points,
                model=best_model,
                ransac_threshold=effective_ransac,
            )
            if M_sub is not None and mask_sub is not None:
                M = M_sub
                working_ref = subpixel_result.refined_points
                final_inlier_mask = mask_sub
                log_step(
                    f"Sub-pixel re-estimation succeeded: {int(np.sum(mask_sub))}/{len(working_src)} "
                    f"inliers retained after sub-pixel geometric verification."
                )
            else:
                log_step(
                    "Sub-pixel re-estimation failed — retaining previous transformation. "
                    "Using spatially-balanced inlier set without sub-pixel mask."
                )
                final_inlier_mask = np.ones(len(working_src), dtype=bool)
        else:
            final_inlier_mask = np.ones(len(working_src), dtype=bool)
            stage_counts["subpixel_candidates"] = 0
            stage_counts["subpixel_verified"] = 0

        final_src = working_src[final_inlier_mask]
        final_ref = working_ref[final_inlier_mask]
        stage_counts["final_inliers"] = min(stage_counts["spatially_balanced"], len(final_src))
        stage_counts["rejected_matches"] = stage_counts["tentative_matches"] - stage_counts["final_inliers"]
        stage_timings["subpixel_refinement_seconds"] = round(time.perf_counter() - t_stage, 4)

        # 8. Image Warping & Product Generation
        t_stage = time.perf_counter()
        log_step("Warping source raster into reference coordinate frame...")
        warped_src = warp_image_to_reference(prep_src, M, (reference.height, reference.width))
        diff_map, mean_diff = compute_difference_map(warped_src, u8_ref)
        alpha_overlay = create_alpha_overlay(warped_src, u8_ref)
        log_step(f"Image warping completed. Mean absolute overlap difference: {mean_diff:.2f}")
        stage_timings["warping_and_products_seconds"] = round(time.perf_counter() - t_stage, 4)

        # 9. Compute Rigorous Metrics and Explainable Quality Score
        t_stage = time.perf_counter()
        final_residuals = calculate_residuals(final_src, final_ref, M)
        final_metrics_mask = np.ones(len(final_src), dtype=bool)

        metrics = compute_registration_metrics(
            final_src, final_ref,
            final_metrics_mask, final_residuals,
            (reference.height, reference.width),
            runtime_seconds=time.perf_counter() - t_start,
            grid_size=spatial_grid_size,
            tentative_match_count=stage_counts["tentative_matches"],
            geometric_inlier_count=stage_counts["geometric_inliers"],
            spatially_balanced_count=stage_counts["spatially_balanced"],
            subpixel_verified_count=stage_counts["subpixel_verified"],
        )

        quality_score = calculate_quality_score(metrics)
        log_step(f"Explainable Quality Score: {quality_score.total_score}/100 ({quality_score.rating_label})")
        stage_timings["metrics_and_evaluation_seconds"] = round(time.perf_counter() - t_stage, 4)
        stage_timings["total_seconds"] = round(time.perf_counter() - t_start, 4)

        # Determine success based on final verified inlier quality
        effective_inliers = stage_counts["final_inliers"]
        effective_inlier_ratio = effective_inliers / max(stage_counts["tentative_matches"], 1)
        registration_reliable = (effective_inliers >= 4) and (effective_inlier_ratio >= 0.15 or effective_inliers >= 8)

        if not registration_reliable:
            rejection_reason = (
                f"Registration rejected: insufficient independent geometric evidence. "
                f"Only {effective_inliers} final verified correspondence(s) remained (minimum 4 required for 2D transformation). "
                f"RMSE is not interpretable and transformation is unvalidated."
            )
            selected_model_display = f"REJECTED (Candidate was {best_model.upper()})"
            model_reason_display = (
                f"Preliminary candidate was {best_model.upper()} ({n_geometric_inliers} consensus inliers), "
                f"but registration was REJECTED: only {effective_inliers} verified inlier(s) survived final verification."
            )
            operating_state = "REJECTED"
            log_step(
                f"WARNING: Registration deemed UNRELIABLE — only {effective_inliers} final verified inliers "
                f"({effective_inlier_ratio*100:.1f}% of {stage_counts['tentative_matches']} tentative). "
                f"Marking as failed to prevent misleading results."
            )
        else:
            rejection_reason = ""
            selected_model_display = best_model
            model_reason_display = reason
            if overlap_rep.level.startswith("Level 3") or overlap_rep.confidence == "UNKNOWN":
                operating_state = "IMAGE_ONLY_EXPLORATORY"
            else:
                operating_state = "VERIFIED_GEOSPATIAL"

        # Assemble final MatchResult
        match_result = MatchResult(
            method=matcher_method,
            source_points=src_pts,
            reference_points=ref_pts,
            confidences=confs,
            inlier_mask=inlier_mask,
            transformation=M,
            transformation_type=selected_model_display,
            residuals=residuals,
            runtime_seconds=match_time,
            diagnostics={
                "model_evaluations": evals,
                "reason": model_reason_display,
                "ransac_method": ransac_method_label,
                "mean_overlap_diff": mean_diff,
                "spatial_uniformity": spatial_result.balanced_uniformity_score if spatial_result else None,
                "subpixel_converged": subpixel_result.n_converged if subpixel_result else None,
                "stage_counts": stage_counts,
                "stage_timings": stage_timings,
                "operating_state": operating_state,
                "rejection_reason": rejection_reason,
            },
        )

        rmse_str = f"{metrics.rmse_px:.3f}px" if metrics.rmse_px is not None else "N/A"
        total_time = time.perf_counter() - t_start
        log_step(
            f"Pipeline finished in {total_time:.3f}s. "
            f"Final Verified Inliers: {stage_counts['final_inliers']}/{stage_counts['tentative_matches']} "
            f"| RMSE: {rmse_str}"
        )

        return PipelineExecutionResult(
            success=registration_reliable,
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
            selected_model=selected_model_display,
            model_selection_reason=model_reason_display,
            model_evaluations=evals,
            preprocessed_source=prep_src,
            preprocessed_reference=prep_ref,
            execution_log=log_messages,
            total_pipeline_time=total_time,
            stage_counts=stage_counts,
            stage_timings=stage_timings,
            operating_state=operating_state,
            rejection_reason=rejection_reason,
        )



