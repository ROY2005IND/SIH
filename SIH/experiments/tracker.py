"""
Experiment Tracker and Run Persistence Engine for MoonFlower AI.
Maintains an append-only CSV benchmark registry and structured JSON run artifacts
for scientific reproducibility and ISRO evaluation auditing.
"""
import csv
import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.hardware import get_hardware_info
from utils.logging import get_logger
from utils.paths import EXPERIMENT_RESULTS_DIR, EXPERIMENT_RUNS_DIR

logger = get_logger("Experiments")

CSV_LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "scenario_name",
    "source_sensor",
    "reference_sensor",
    "preprocessing",
    "matcher",
    "selected_model",
    "tentative_matches",
    "inlier_count",
    "inlier_ratio_pct",
    "rmse_px",
    "median_res_px",
    "coverage_pct",
    "uniformity_score",
    "subpixel_converged",
    "quality_score",
    "rating",
    "runtime_s",
    "notes",
]

class ExperimentTracker:
    """
    Central experiment tracking system for MoonFlower AI.
    Persists evaluation metrics, algorithm configurations, and hardware telemetry.
    """

    def __init__(
        self,
        runs_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.runs_dir = Path(runs_dir) if runs_dir else EXPERIMENT_RUNS_DIR
        self.results_dir = Path(results_dir) if results_dir else EXPERIMENT_RESULTS_DIR
        self.csv_path = self.results_dir / "experiments_log.csv"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Create experiments_log.csv with standardized column schema if not present."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            try:
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_LOG_COLUMNS)
            except Exception as e:
                logger.error(f"Failed to initialize CSV log header at {self.csv_path}: {e}")

    def record_run(
        self,
        result: Any = None,
        scenario_name: str = "custom",
        source_sensor: str = "Unknown",
        reference_sensor: str = "Unknown",
        preprocessing_method: str = "clahe",
        matcher_method: str = "sift",
        notes: str = "",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a registration experiment execution into JSON and CSV registries.

        Parameters
        ----------
        result : PipelineExecutionResult or None
            Result object from RegistrationPipeline.run().
        scenario_name : str
            Identifier of the evaluation scenario or dataset pair.
        source_sensor : str
            Name of source sensor (e.g. OHRC, TMC-2, IIRS, LRO NAC).
        reference_sensor : str
            Name of reference sensor (e.g. TMC-2, SELENE TC).
        preprocessing_method : str
            Name of radiometric/edge preprocessing method used.
        matcher_method : str
            Feature matching algorithm used.
        notes : str
            Human-readable annotations or experiment objectives.
        custom_metadata : dict, optional
            Arbitrary key-value metadata to store in JSON run file.

        Returns
        -------
        str
            Unique run ID (e.g. run_20260907_143000_a1b2c3d4).
        """
        now = datetime.now(timezone.utc)
        timestamp_iso = now.isoformat()
        short_id = uuid.uuid4().hex[:8]
        run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{short_id}"

        # Default metric fields
        selected_model = "none"
        tentative_matches = 0
        inlier_count = 0
        inlier_ratio_pct = 0.0
        rmse_px: Optional[float] = None
        median_res_px: Optional[float] = None
        max_res_px: Optional[float] = None
        coverage_pct: Optional[float] = None
        uniformity_score: Optional[float] = None
        subpixel_converged = 0
        quality_score_val = 0.0
        rating_label = "FAILED"
        runtime_s = 0.0
        success = False

        json_data: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp_iso,
            "scenario_name": scenario_name,
            "source_sensor": source_sensor,
            "reference_sensor": reference_sensor,
            "preprocessing": preprocessing_method,
            "matcher": matcher_method,
            "notes": notes,
            "hardware": get_hardware_info(),
            "custom_metadata": custom_metadata or {},
        }

        # Extract from PipelineExecutionResult if provided
        if result is not None:
            success = bool(getattr(result, "success", False))
            selected_model = str(getattr(result, "selected_model", "none"))
            runtime_s = round(float(getattr(result, "total_pipeline_time", 0.0)), 4)

            # Match Result details
            match_res = getattr(result, "match_result", None)
            if match_res is not None:
                if hasattr(match_res, "method"):
                    matcher_method = match_res.method
                tentative_matches = len(match_res.source_points) if match_res.source_points is not None else 0
                inlier_count = match_res.n_inliers
                inlier_ratio_pct = round(match_res.inlier_ratio * 100.0, 2)
                rmse_px = round(float(match_res.rmse), 4) if match_res.rmse is not None else None

                if match_res.transformation is not None:
                    json_data["transformation_matrix"] = match_res.transformation.tolist()

            # Metrics
            metrics = getattr(result, "metrics", None)
            if metrics is not None:
                tentative_matches = getattr(metrics, "tentative_matches", tentative_matches)
                inlier_count = getattr(metrics, "inlier_count", inlier_count)
                inlier_ratio_pct = getattr(metrics, "inlier_ratio_pct", inlier_ratio_pct)
                rmse_px = getattr(metrics, "rmse_px", rmse_px)
                median_res_px = getattr(metrics, "median_residual_px", None)
                max_res_px = getattr(metrics, "max_inlier_error_px", getattr(metrics, "p90_residual_px", None))
                coverage_pct = getattr(metrics, "spatial_coverage_pct", None)
                uniformity_score = getattr(metrics, "spatial_uniformity_score", None)
                json_data["metrics"] = asdict(metrics) if is_dataclass(metrics) else str(metrics)

            # Quality Score
            qs = getattr(result, "quality_score", None)
            if qs is not None:
                quality_score_val = qs.total_score
                rating_label = qs.rating_label
                json_data["quality_score"] = asdict(qs) if is_dataclass(qs) else str(qs)

            # Sub-pixel result
            sub_res = getattr(result, "subpixel_result", None)
            if sub_res is not None:
                subpixel_converged = sub_res.n_converged
                json_data["subpixel"] = {
                    "converged": sub_res.n_converged,
                    "mean_offset_px": sub_res.mean_offset_px,
                    "max_offset_px": sub_res.max_offset_px,
                }

            # Spatial result
            spat_res = getattr(result, "spatial_result", None)
            if spat_res is not None:
                json_data["spatial"] = {
                    "grid_size": spat_res.grid_size,
                    "retained_points": len(spat_res.balanced_source_points),
                    "raw_uniformity": spat_res.raw_uniformity_score,
                    "balanced_uniformity": spat_res.balanced_uniformity_score,
                }

            # Overlap & Image Quality
            ov_rep = getattr(result, "overlap_report", None)
            if ov_rep is not None:
                json_data["overlap"] = asdict(ov_rep) if is_dataclass(ov_rep) else str(ov_rep)

            src_q = getattr(result, "source_quality", None)
            if src_q is not None:
                json_data["source_quality"] = asdict(src_q) if is_dataclass(src_q) else str(src_q)

            ref_q = getattr(result, "reference_quality", None)
            if ref_q is not None:
                json_data["reference_quality"] = asdict(ref_q) if is_dataclass(ref_q) else str(ref_q)

            # Execution log
            json_data["execution_log"] = getattr(result, "execution_log", [])

        json_data["success"] = success
        json_data["summary"] = {
            "selected_model": selected_model,
            "tentative_matches": tentative_matches,
            "inlier_count": inlier_count,
            "inlier_ratio_pct": inlier_ratio_pct,
            "rmse_px": rmse_px,
            "median_res_px": median_res_px,
            "coverage_pct": coverage_pct,
            "uniformity_score": uniformity_score,
            "subpixel_converged": subpixel_converged,
            "quality_score": quality_score_val,
            "rating": rating_label,
            "runtime_s": runtime_s,
        }

        # 1. Write detailed JSON run file
        json_path = self.runs_dir / f"{run_id}.json"
        try:
            with open(json_path, mode="w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"Recorded detailed experiment run artifact: {json_path}")
        except Exception as e:
            logger.error(f"Failed to persist run JSON to {json_path}: {e}")

        # 2. Append row to CSV log
        csv_row = [
            run_id,
            timestamp_iso,
            scenario_name,
            source_sensor,
            reference_sensor,
            preprocessing_method,
            matcher_method,
            selected_model,
            tentative_matches,
            inlier_count,
            inlier_ratio_pct,
            rmse_px if rmse_px is not None else "",
            median_res_px if median_res_px is not None else "",
            coverage_pct if coverage_pct is not None else "",
            uniformity_score if uniformity_score is not None else "",
            subpixel_converged,
            quality_score_val,
            rating_label,
            runtime_s,
            notes.replace("\n", " "),
        ]

        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)
            logger.info(f"Appended experiment run summary to CSV log: {self.csv_path}")
        except Exception as e:
            logger.error(f"Failed to append to CSV log: {e}")

        return run_id

    def get_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve experiment run history as a pandas DataFrame.

        Parameters
        ----------
        limit : int, optional
            Maximum number of recent runs to return.

        Returns
        -------
        pd.DataFrame
            DataFrame containing history log or empty DataFrame if no records.
        """
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return pd.DataFrame(columns=CSV_LOG_COLUMNS)

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return pd.DataFrame(columns=CSV_LOG_COLUMNS)
            # Reverse order so latest is on top
            df = df.iloc[::-1].reset_index(drop=True)
            if limit is not None and limit > 0:
                df = df.head(limit)
            return df
        except Exception as e:
            logger.error(f"Failed to read experiment log CSV: {e}")
            return pd.DataFrame(columns=CSV_LOG_COLUMNS)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete JSON artifact for a specific run ID.

        Parameters
        ----------
        run_id : str
            The identifier of the run.

        Returns
        -------
        dict or None
            Parsed JSON artifact or None if not found.
        """
        clean_id = Path(run_id).stem
        json_path = self.runs_dir / f"{clean_id}.json"
        if not json_path.exists():
            logger.warning(f"Run artifact not found: {json_path}")
            return None

        try:
            with open(json_path, mode="r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load run artifact from {json_path}: {e}")
            return None

    def list_runs(self) -> List[str]:
        """
        List all recorded run IDs sorted by creation time descending.

        Returns
        -------
        list of str
        """
        files = sorted(
            self.runs_dir.glob("run_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [f.stem for f in files]

    def export_summary_csv(self, dest_path: Union[Path, str]) -> Path:
        """
        Export the current experiments CSV log to an external destination.

        Parameters
        ----------
        dest_path : Path or str
            Destination filepath.

        Returns
        -------
        Path
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_history()
        df.to_csv(dest, index=False)
        logger.info(f"Exported experiments history to {dest}")
        return dest
