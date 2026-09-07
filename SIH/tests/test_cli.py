"""Unit tests for MoonFlower command-line interface (cli.py)."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_cli_help():
    """Verify cli.py --help returns exit code 0 and displays usage."""
    proc = subprocess.run(
        [sys.executable, "cli.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "MoonFlower AI" in proc.stdout
    assert "--scenario" in proc.stdout
    assert "--prep" in proc.stdout

def test_cli_list_scenarios():
    """Verify cli.py --list-scenarios outputs catalog."""
    proc = subprocess.run(
        [sys.executable, "cli.py", "--list-scenarios"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "scenario_1_small_scale" in proc.stdout
    assert "scenario_5_illumination_shift" in proc.stdout

def test_cli_json_execution_on_scenario(tmp_path: Path):
    """Verify cli.py executes on scenario and outputs parseable JSON payload."""
    export_dir = tmp_path / "cli_export"
    proc = subprocess.run(
        [
            sys.executable, "cli.py",
            "--scenario", "scenario_1_small_scale",
            "--prep", "clahe",
            "--matcher", "sift",
            "--export-dir", str(export_dir),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    # Locate JSON block in stdout
    stdout_clean = proc.stdout.strip()
    json_start = stdout_clean.find("{")
    assert json_start != -1
    payload = json.loads(stdout_clean[json_start:])

    assert payload["success"] is True
    assert payload["selected_model"] in ["similarity", "affine", "homography"]
    assert payload["matches"]["inliers"] > 5
    assert payload["quality_score"]["total_score"] > 60
    assert export_dir.exists()
