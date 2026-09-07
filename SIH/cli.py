"""
MoonFlower AI — Scientific Command-Line Interface (CLI).
Enables headless, automated, and batch planetary image registration.
Suitable for HPC clusters, automated ground station ingest, and headless evaluation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.scenarios import SCENARIOS_CATALOG, get_scenario_by_id
from evaluation.benchmark import run_comparative_benchmark
from experiments.tracker import ExperimentTracker
from pipeline.export import export_registration_package
from pipeline.registration_pipeline import RegistrationPipeline
from preprocessing.loader import load_lunar_image
from utils.hardware import get_hardware_info
from utils.logging import get_logger

logger = get_logger("CLI")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moonflower",
        description="MoonFlower AI: Multi-Modal, Sun-Angle and Scale-Invariant Lunar Image Registration (ISRO SIH #26166)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group_input = parser.add_argument_group("Input Selection")
    group_input.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a controlled synthetic benchmark scenario (e.g. 'scenario_1_small_scale', 'scenario_5_illumination_shift').",
    )
    group_input.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available controlled benchmark scenarios and exit.",
    )
    group_input.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to source observed lunar raster (GeoTIFF, TIFF, PNG, JPEG).",
    )
    group_input.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference lunar base raster (GeoTIFF, TIFF, PNG, JPEG).",
    )

    group_pipeline = parser.add_argument_group("Algorithmic Parameters")
    group_pipeline.add_argument(
        "--prep",
        choices=["clahe", "gradient", "original", "phase", "edge"],
        default="clahe",
        help="Radiometric/edge preprocessing representation.",
    )
    group_pipeline.add_argument(
        "--matcher",
        choices=["hybrid", "sift", "akaze"],
        default="hybrid",
        help="Feature matching engine ('hybrid' fuses SIFT and learned features).",
    )
    group_pipeline.add_argument(
        "--ransac-thresh",
        type=float,
        default=3.0,
        help="RANSAC geometric verification inlier threshold in pixels.",
    )
    group_pipeline.add_argument(
        "--spatial-grid",
        type=int,
        choices=[4, 6, 8],
        default=6,
        help="Grid partitioning size for the Adaptive Spatial Match Balancer.",
    )
    group_pipeline.add_argument(
        "--no-spatial-balancing",
        action="store_true",
        help="Disable the Adaptive Spatial Match Balancer.",
    )
    group_pipeline.add_argument(
        "--no-subpixel",
        action="store_true",
        help="Disable 2D parabolic NCC sub-pixel peak refinement.",
    )

    group_output = parser.add_argument_group("Output & Reporting")
    group_output.add_argument(
        "--benchmark",
        action="store_true",
        help="Execute progressive comparative benchmark (SIFT vs Balancer vs Subpixel vs Full).",
    )
    group_output.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Directory to save full export package (CSV, JSON, Warped Image, Markdown report).",
    )
    group_output.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result to stdout.",
    )
    group_output.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable informational banners.",
    )

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # List scenarios if requested
    if args.list_scenarios:
        print("\nAvailable Controlled Lunar Benchmark Scenarios:")
        for s_id, title in SCENARIOS_CATALOG:
            print(f"  - {s_id:<35} : {title}")
        print()
        return 0

    # Determine input images
    scenario_meta = None
    if args.scenario:
        scenario_meta = get_scenario_by_id(args.scenario, size=512, seed=42)
        source = scenario_meta.source_image
        reference = scenario_meta.reference_image
        dataset_name = scenario_meta.title
    elif args.source and args.reference:
        src_path = Path(args.source)
        ref_path = Path(args.reference)
        if not src_path.exists():
            print(f"Error: Source file not found: {src_path}", file=sys.stderr)
            return 1
        if not ref_path.exists():
            print(f"Error: Reference file not found: {ref_path}", file=sys.stderr)
            return 1
        source = load_lunar_image(src_path)
        reference = load_lunar_image(ref_path)
        dataset_name = f"{src_path.name} -> {ref_path.name}"
    else:
        # Default to Scenario 1 if no arguments passed
        scenario_meta = get_scenario_by_id("scenario_1_small_scale", size=512, seed=42)
        source = scenario_meta.source_image
        reference = scenario_meta.reference_image
        dataset_name = scenario_meta.title
        if not args.quiet and not args.json:
            print("[INFO] No input specified. Defaulting to Scenario 1: Small Scale Variation (1.2x).")

    if not args.quiet and not args.json:
        hw = get_hardware_info()
        print("=" * 72)
        print(" 🛰️  MoonFlower AI — SCIENTIFIC MISSION REGISTRATION ENGINE")
        print(f" Target: {dataset_name}")
        print(f" Compute: {hw['cpu_threads']} Threads | Device: {hw['device'].upper()} ({hw['gpu_name']})")
        print(f" Config: Prep={args.prep.upper()} | Matcher={args.matcher.upper()} | RANSAC={args.ransac_thresh}px")
        print("=" * 72)

    # Progressive Benchmark Execution
    if args.benchmark:
        if not args.quiet and not args.json:
            print("\n[BENCHMARK] Executing progressive comparative benchmark suite...")
        df_bench = run_comparative_benchmark(
            source, reference,
            preprocessing=args.prep,
            grid_size=args.spatial_grid,
        )
        if args.json:
            print(df_bench.to_json(orient="records", indent=2))
        else:
            print("\n" + df_bench.to_string(index=False) + "\n")
        return 0

    # Pipeline Execution
    pipeline = RegistrationPipeline()
    result = pipeline.run(
        source=source,
        reference=reference,
        preprocessing_method=args.prep,
        matcher_method=args.matcher,
        ransac_threshold=args.ransac_thresh,
        enable_spatial_balancing=not args.no_spatial_balancing,
        spatial_grid_size=args.spatial_grid,
        enable_subpixel=not args.no_subpixel,
    )

    # Persist in Experiment Registry
    tracker = ExperimentTracker()
    run_id = tracker.record_run(
        result=result,
        scenario_name=dataset_name,
        source_sensor=source.instrument if source.instrument != "Not available" else "Source",
        reference_sensor=reference.instrument if reference.instrument != "Not available" else "Reference",
        preprocessing_method=args.prep,
        matcher_method=args.matcher,
        notes="Executed via MoonFlower Headless CLI",
    )

    # Export Package if requested
    export_paths = {}
    if args.export_dir:
        out_dir = Path(args.export_dir)
        export_paths = export_registration_package(result, source, reference, export_dir=out_dir)
        if not args.quiet and not args.json:
            print(f"[EXPORT] Saved scientific artifacts to: {out_dir}")

    # JSON output
    if args.json:
        payload = {
            "success": result.success,
            "run_id": run_id,
            "scenario": dataset_name,
            "selected_model": result.selected_model,
            "model_selection_reason": result.model_selection_reason,
            "total_time_seconds": round(result.total_pipeline_time, 4),
            "matches": {
                "tentative": result.metrics.tentative_matches if result.metrics else 0,
                "inliers": result.metrics.inlier_count if result.metrics else 0,
                "inlier_ratio_pct": result.metrics.inlier_ratio_pct if result.metrics else 0.0,
            },
            "precision": {
                "rmse_px": result.metrics.rmse_px if result.metrics else None,
                "median_residual_px": result.metrics.median_residual_px if result.metrics else None,
                "spatial_coverage_pct": result.metrics.spatial_coverage_pct if result.metrics else 0.0,
                "spatial_uniformity_score": result.metrics.spatial_uniformity_score if result.metrics else 0.0,
            },
            "subpixel": {
                "converged": result.subpixel_result.n_converged if result.subpixel_result else 0,
                "mean_offset_px": result.subpixel_result.mean_offset_px if result.subpixel_result else 0.0,
            },
            "quality_score": {
                "total_score": result.quality_score.total_score if result.quality_score else 0.0,
                "rating": result.quality_score.rating_label if result.quality_score else "FAILED",
            },
            "transformation_matrix": result.match_result.transformation.tolist() if result.match_result.transformation is not None else None,
            "export_files": {k: str(v) for k, v in export_paths.items()},
        }
        print(json.dumps(payload, indent=2))
        return 0 if result.success else 1

    # Human-readable stdout
    if not args.quiet:
        if result.success and result.metrics and result.quality_score:
            m = result.metrics
            q = result.quality_score
            print("\n" + "-" * 72)
            print(f" ✓ REGISTRATION STATUS: SUCCESS (Run ID: {run_id})")
            print(f"   Model Selected:     {result.selected_model.upper()} ({result.model_selection_reason})")
            print(f"   Inlier RMSE:        {m.rmse_px:.3f} px (Median: {m.median_residual_px:.3f} px)")
            print(f"   Inliers / Tentative:{m.inlier_count} / {m.tentative_matches} ({m.inlier_ratio_pct:.1f}%)")
            print(f"   Spatial Uniformity: {m.spatial_uniformity_score:.1f} / 100 ({args.spatial_grid}x{args.spatial_grid} Grid)")
            print(f"   Spatial Coverage:   {m.spatial_coverage_pct:.1f}%")
            if result.subpixel_result:
                print(f"   Sub-Pixel Precision:{result.subpixel_result.n_converged} points refined (Mean offset: {result.subpixel_result.mean_offset_px:.3f} px)")
            print(f"   Quality Score:      {q.total_score}/100 [{q.rating_label}]")
            print(f"   Total Pipeline Time:{result.total_pipeline_time:.3f}s")
            print("-" * 72 + "\n")
        else:
            print("\n❌ REGISTRATION FAILED: Unable to verify sufficient spatial correspondences.\n", file=sys.stderr)
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
