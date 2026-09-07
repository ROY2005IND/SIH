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
    candidate_models: Sequence[TransformationModel] = ("similarity", "affine", "homography"),
    ransac_threshold: float = 3.0,
) -> tuple[TransformationModel, np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Evaluate candidate geometric models and select the most appropriate transformation.
    Returns:
        best_model: selected model name
        best_M: estimated transformation matrix
        best_mask: inlier boolean mask
        best_residuals: per-point transfer residuals
        selection_reason: explainable scientific justification
        evaluations_dict: performance comparison dictionary
    """
    evals: list[ModelEvaluation] = []
    n_pts = len(source_points)

    for model in candidate_models:
        M, mask, res = estimate_robust_transformation(
            source_points, reference_points,
            model=model,
            ransac_threshold=ransac_threshold,
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

    if not evals:
        # Complete failure to fit any model
        raise RuntimeError("Failed to fit any geometric transformation model to correspondence points.")

    # Sort candidates by ascending complexity (DoF)
    evals.sort(key=lambda x: x.dof)

    # Baseline is the simplest successful model
    best = evals[0]
    reason = f"Selected {best.model.upper()} ({best.dof} DoF) as the simplest adequate model with RMSE={best.inlier_rmse:.2f}px and {best.inlier_count} inliers."

    # Compare with higher-order models: only upgrade if RMSE improves by at least 15%
    for candidate in evals[1:]:
        rmse_improvement = (best.inlier_rmse - candidate.inlier_rmse) / max(best.inlier_rmse, 1e-6)
        inlier_gain = candidate.inlier_count - best.inlier_count

        if (rmse_improvement > 0.15 and candidate.inlier_rmse < best.inlier_rmse) or inlier_gain > 5:
            reason = (
                f"Selected {candidate.model.upper()} ({candidate.dof} DoF) over {best.model.upper()} ({best.dof} DoF): "
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
        }
        for e in evals
    }

    assert best.transformation is not None
    assert best.inlier_mask is not None
    assert best.residuals is not None

    return best.model, best.transformation, best.inlier_mask, best.residuals, reason, eval_summary
