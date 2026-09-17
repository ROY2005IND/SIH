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


def _run_scenario(scenario_id: str, size: int = 256, **kwargs):
    """Run a pipeline scenario and return the result."""
    scenario_data = get_scenario_by_id(scenario_id, size=size, seed=42)
    pipeline = RegistrationPipeline()
    return pipeline.run(
        source=scenario_data.source_image,
        reference=scenario_data.reference_image,
        preprocessing_method="clahe",
        matcher_method="sift",
        **kwargs,
    )


class TestStageCounts:
    """Verify that point counts decrease monotonically through pipeline stages."""

    def test_stage_counts_present(self):
        result = _run_scenario("scenario_1_small_scale")
        assert result.success is True
        sc = result.stage_counts
        required_keys = {
            "tentative_matches",
            "geometric_inliers",
            "spatially_balanced",
            "final_inliers",
            "rejected_matches",
        }
        assert required_keys.issubset(sc.keys())

    def test_stage_counts_monotonic(self):
        result = _run_scenario("scenario_1_small_scale")
        sc = result.stage_counts
        assert sc["tentative_matches"] >= sc["geometric_inliers"] >= 0
        assert sc["geometric_inliers"] >= sc["spatially_balanced"] >= 0
        assert sc["spatially_balanced"] >= sc["final_inliers"] >= 0

    def test_rejected_equals_tentative_minus_final(self):
        result = _run_scenario("scenario_1_small_scale")
        sc = result.stage_counts
        assert sc["rejected_matches"] == sc["tentative_matches"] - sc["final_inliers"]


class TestMetricsConsistency:
    """Verify that metrics use final verified inliers, not all working points."""

    def test_metrics_inlier_count_matches_stage(self):
        result = _run_scenario("scenario_1_small_scale")
        assert result.metrics is not None
        assert result.metrics.inlier_count == result.stage_counts["final_inliers"]
        assert result.metrics.final_inlier_count == result.stage_counts["final_inliers"]

    def test_metrics_stage_fields_populated(self):
        result = _run_scenario("scenario_1_small_scale")
        m = result.metrics
        assert m.geometric_inlier_count > 0
        assert m.spatially_balanced_count > 0
        assert m.final_inlier_count > 0
        assert m.tentative_matches > 0
        assert m.rejected_match_count >= 0

    def test_inlier_ratio_against_tentative(self):
        result = _run_scenario("scenario_1_small_scale")
        m = result.metrics
        expected_ratio = round((m.final_inlier_count / max(m.tentative_matches, 1)) * 100.0, 2)
        assert m.inlier_ratio_pct == expected_ratio


class TestSubPixelReEstimation:
    """Verify sub-pixel refinement mask handling."""

    def test_subpixel_enabled_populates_result(self):
        result = _run_scenario("scenario_1_small_scale", enable_subpixel=True)
        assert result.subpixel_result is not None
        assert result.subpixel_result.n_converged > 0

    def test_subpixel_disabled_skips_refinement(self):
        result = _run_scenario("scenario_1_small_scale", enable_subpixel=False)
        assert result.subpixel_result is None
        sc = result.stage_counts
        assert sc["subpixel_candidates"] == 0
        assert sc["subpixel_verified"] == 0
        assert sc["final_inliers"] == sc["spatially_balanced"]


class TestSpatialBalancingInteraction:
    """Verify that spatial balancing and sub-pixel interact correctly."""

    def test_no_spatial_balancing(self):
        result = _run_scenario("scenario_1_small_scale", enable_spatial_balancing=False)
        assert result.success is True
        sc = result.stage_counts
        assert sc["spatially_balanced"] == sc["geometric_inliers"]


class TestMatchResultConsistency:
    """Verify MatchResult stores consistent data."""

    def test_match_result_has_diagnostics(self):
        result = _run_scenario("scenario_1_small_scale")
        diag = result.match_result.diagnostics
        assert "stage_counts" in diag
        assert "ransac_method" in diag
        assert diag["ransac_method"] in ("USAC_MAGSAC", "OpenCV_RANSAC")

