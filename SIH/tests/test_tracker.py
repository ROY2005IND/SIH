"""Unit and integration tests for ExperimentTracker."""
from pathlib import Path
import pytest
from experiments.tracker import ExperimentTracker, CSV_LOG_COLUMNS
from pipeline.registration_pipeline import RegistrationPipeline
from demo.scenarios import get_scenario_by_id

def test_experiment_tracker_basic_recording(tmp_path: Path):
    """Verify manual run recording creates JSON artifact and appends to CSV."""
    runs_dir = tmp_path / "runs"
    results_dir = tmp_path / "results"
    tracker = ExperimentTracker(runs_dir=runs_dir, results_dir=results_dir)

    run_id = tracker.record_run(
        result=None,
        scenario_name="Test Scenario",
        source_sensor="OHRC",
        reference_sensor="TMC-2",
        preprocessing_method="clahe",
        matcher_method="sift",
        notes="Testing basic tracker persistence",
    )

    assert run_id.startswith("run_")
    assert (runs_dir / f"{run_id}.json").exists()
    assert tracker.csv_path.exists()

    df = tracker.get_history()
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == run_id
    assert df.iloc[0]["source_sensor"] == "OHRC"
    assert df.iloc[0]["reference_sensor"] == "TMC-2"

    detail = tracker.get_run(run_id)
    assert detail is not None
    assert detail["scenario_name"] == "Test Scenario"
    assert "hardware" in detail

def test_experiment_tracker_with_pipeline_result(tmp_path: Path):
    """Verify recording directly from a PipelineExecutionResult extracts metrics correctly."""
    runs_dir = tmp_path / "runs"
    results_dir = tmp_path / "results"
    tracker = ExperimentTracker(runs_dir=runs_dir, results_dir=results_dir)

    scenario = get_scenario_by_id("scenario_1_small_scale", size=128, seed=42)
    pipeline = RegistrationPipeline()
    res = pipeline.run(scenario.source_image, scenario.reference_image)

    run_id = tracker.record_run(
        result=res,
        scenario_name="Scenario 1 Synthetic",
        source_sensor="OHRC-Sim",
        reference_sensor="TMC-Sim",
        notes="Pipeline execution record test",
    )

    df = tracker.get_history()
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == run_id
    assert df.iloc[0]["selected_model"] in ["similarity", "affine", "homography"]
    assert float(df.iloc[0]["inlier_count"]) > 5
    assert float(df.iloc[0]["quality_score"]) > 0

    run_json = tracker.get_run(run_id)
    assert run_json is not None
    assert "metrics" in run_json
    assert "quality_score" in run_json
    assert run_json["success"] is True

def test_experiment_tracker_export(tmp_path: Path):
    """Verify exporting summary CSV to external path."""
    tracker = ExperimentTracker(runs_dir=tmp_path / "runs", results_dir=tmp_path / "results")
    tracker.record_run(scenario_name="A")
    tracker.record_run(scenario_name="B")

    export_dest = tmp_path / "exports" / "custom_summary.csv"
    exported = tracker.export_summary_csv(export_dest)
    assert exported.exists()
    assert exported.stat().st_size > 50
