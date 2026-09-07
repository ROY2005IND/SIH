"""Integration test suite for RegistrationPipeline orchestrator."""
import pytest
from demo.scenarios import get_scenario_by_id
from pipeline.registration_pipeline import RegistrationPipeline

def test_pipeline_on_scenario_1_small_scale():
    """Verify complete end-to-end registration pipeline on Scenario 1 (Small scale)."""
    scenario_data = get_scenario_by_id("scenario_1_small_scale", size=256, seed=42)
    pipeline = RegistrationPipeline()

    result = pipeline.run(
        source=scenario_data.source_image,
        reference=scenario_data.reference_image,
        preprocessing_method="clahe",
        matcher_method="sift",
    )

    assert result.success is True
    assert result.match_result.n_inliers > 10
    assert result.match_result.inlier_ratio > 0.50
    assert result.match_result.rmse is not None
    assert result.match_result.rmse < 2.5  # Sub-pixel to low pixel accuracy
    assert result.registered_image is not None
    assert result.difference_map is not None
    assert result.selected_model in ["similarity", "affine", "homography"]
    assert len(result.execution_log) >= 5

def test_pipeline_on_scenario_3_rotation():
    """Verify complete end-to-end registration on Scenario 3 (25 deg rotation + translation)."""
    scenario_data = get_scenario_by_id("scenario_3_rotation_translation", size=256, seed=42)
    pipeline = RegistrationPipeline()

    result = pipeline.run(
        source=scenario_data.source_image,
        reference=scenario_data.reference_image,
        preprocessing_method="clahe",
        matcher_method="sift",
    )

    assert result.success is True
    assert result.match_result.n_inliers > 8
    assert result.match_result.rmse is not None
    assert result.match_result.rmse < 3.0

def test_pipeline_with_gradient_preprocessing():
    """Verify gradient representation pipeline execution."""
    scenario_data = get_scenario_by_id("scenario_1_small_scale", size=256, seed=42)
    pipeline = RegistrationPipeline()

    result = pipeline.run(
        source=scenario_data.source_image,
        reference=scenario_data.reference_image,
        preprocessing_method="gradient",
        matcher_method="sift",
    )

    assert result.success is True
    assert result.registered_image is not None
