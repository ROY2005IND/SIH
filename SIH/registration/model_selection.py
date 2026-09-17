"""
Automated Transformation Model Selection Engine.
Selects the simplest adequate geometric model (Translation vs Similarity vs Affine vs Homography)
based on residual analysis and degrees-of-freedom penalization (Occam's razor).
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .transformation import TransformationModel, MODEL_DOF
from .ransac import estimate_robust_transformation

@dataclass
class ModelEvaluation:
    """Evaluation metrics for a candidate geometric transformation model."""
    model: TransformationModel
    dof: int
    transformation: np.ndarray | None
    inlier_mask: np.ndarray | None
    residuals: np.ndarray | None
    inlier_count: int
    inlier_ratio: float
    inlier_rmse: float
    bic_score: float

def select_best_transformation_model(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    candidate_models: Sequence[TransformationModel] = ("translation", "similarity", "affine", "homography"),
    ransac_threshold: float = 3.0,
) -> tuple[TransformationModel, np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Evaluate candidate geometric models and select the most appropriate transformation.
    Includes progressive threshold search and robust median fallback.
    """
    evals: list[ModelEvaluation] = []
    n_pts = len(source_points)

    if n_pts == 0:
        raise RuntimeError("No correspondence points available for geometric model selection.")

    # Progressive threshold search to handle initial scale/resolution noise on large rasters
    thresholds_to_try = [ransac_threshold, max(ransac_threshold * 2.0, 6.0), max(ransac_threshold * 4.0, 12.0)]

    for thresh in thresholds_to_try:
        evals = []
        for model in candidate_models:
            M, mask, res = estimate_robust_transformation(
                source_points, reference_points,
                model=model,
                ransac_threshold=thresh,
            )
            if M is not None and mask is not None and res is not None and np.sum(mask) >= MODEL_DOF[model]:
                inliers = int(np.sum(mask))
                ratio = float(inliers / max(n_pts, 1))
                inlier_res = res[mask]
                rmse = float(np.sqrt(np.mean(inlier_res ** 2))) if len(inlier_res) > 0 else 999.0
                dof = MODEL_DOF[model]

                # Bayesian Information Criterion approximation: BIC = N * ln(MSE) + k * ln(N)
                mse = max(rmse ** 2, 1e-6)
                bic = float(inliers * np.log(mse) + dof * np.log(max(inliers, 2)))

                evals.append(ModelEvaluation(
                    model=model,
                    dof=dof,
                    transformation=M,
                    inlier_mask=mask,
                    residuals=res,
                    inlier_count=inliers,
                    inlier_ratio=ratio,
                    inlier_rmse=rmse,
                    bic_score=bic,
                ))

        if evals:
            break

    if not evals:
        # Check if a relaxed translation model can find genuine consensus (at least 4 inliers)
        deltas = reference_points - source_points
        med_delta = np.median(deltas, axis=0)
        res = np.linalg.norm(deltas - med_delta, axis=1)
        mask = res <= max(ransac_threshold * 3.0, 10.0)
        n_relaxed_inliers = int(np.sum(mask))

        if n_relaxed_inliers < 4:
            # DO NOT fabricate an all-ones mask: fail honestly
            raise RuntimeError(
                f"Geometric verification failed: no consensus could be found among {n_pts} correspondence candidates. "
                "The images may be non-overlapping, poorly conditioned, or contain incompatible geometric distortion."
            )

        ref_delta = np.mean(deltas[mask], axis=0)
        M_fallback = np.array([[1.0, 0.0, ref_delta[0]], [0.0, 1.0, ref_delta[1]]], dtype=np.float64)
        inlier_res = res[mask]
        rmse = float(np.sqrt(np.mean(inlier_res ** 2))) if len(inlier_res) > 0 else 10.0

        return (
            "translation",
            M_fallback,
            mask,
            res,
            f"Relaxed TRANSLATION model estimated via median delta [dx={ref_delta[0]:.1f}, dy={ref_delta[1]:.1f}] px ({n_relaxed_inliers}/{n_pts} points within tolerance).",
            {"translation": {"dof": 2, "inliers": n_relaxed_inliers, "rmse": rmse, "method": "Median_RANSAC"}}
        )

    # Sort candidates by ascending complexity (DoF)
    evals.sort(key=lambda x: x.dof)

    # Baseline is the simplest successful model
    best = evals[0]
    reason = f"Selected {best.model.upper()} ({best.dof} DoF) as the simplest adequate model with RMSE={best.inlier_rmse:.2f}px and {best.inlier_count} inliers."

    # Compare with higher-order models using BIC (Bayesian Information Criterion) as primary Occam's razor rule:
    # A lower BIC score indicates a statistically superior model that justifies higher degrees of freedom.
    # A BIC difference > 2.0 represents positive evidence for model upgrade (Kass & Raftery 1995).
    for candidate in evals[1:]:
        rmse_improvement = (best.inlier_rmse - candidate.inlier_rmse) / max(best.inlier_rmse, 1e-6)
        inlier_gain = candidate.inlier_count - best.inlier_count
        bic_improvement = best.bic_score - candidate.bic_score

        # Primary criterion: BIC improvement > 2.0 or significant RMSE improvement (>15%) with positive inlier gain
        if bic_improvement > 2.0 or (rmse_improvement > 0.15 and inlier_gain >= 0 and bic_improvement > -5.0) or inlier_gain > 5:
            reason = (
                f"Selected {candidate.model.upper()} ({candidate.dof} DoF) over {best.model.upper()} ({best.dof} DoF) via BIC analysis: "
                f"BIC improved by {bic_improvement:.1f} ({best.bic_score:.1f} -> {candidate.bic_score:.1f}) | "
                f"RMSE improved by {rmse_improvement * 100.0:.1f}% ({best.inlier_rmse:.2f}px -> {candidate.inlier_rmse:.2f}px) "
                f"with {inlier_gain:+d} additional inliers."
            )
            best = candidate

    eval_summary = {
        e.model: {
            "dof": e.dof,
            "inliers": e.inlier_count,
            "inlier_ratio": round(e.inlier_ratio, 3),
            "rmse_px": round(e.inlier_rmse, 3),
            "bic": round(e.bic_score, 1),
            "method": "USAC_MAGSAC" if e.model == "homography" else ("Median_RANSAC" if e.model == "translation" else "OpenCV_RANSAC"),
        }
        for e in evals
    }

    assert best.transformation is not None
    assert best.inlier_mask is not None
    assert best.residuals is not None

    return best.model, best.transformation, best.inlier_mask, best.residuals, reason, eval_summary

