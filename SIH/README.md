# MoonFlower AI
### Multi-Modal, Sun-Angle & Scale-Invariant Lunar Image Correspondence and Sub-Pixel Registration

**Smart India Hackathon (SIH) Problem Statement #26166**  
**Issuing Organization:** Indian Space Research Organisation (ISRO)

---

## 1. Executive Summary

Lunar surface imagery acquired by planetary orbiters (**Chandrayaan-2 OHRC, TMC-2, IIRS**, and reference missions **LRO NAC, SELENE/Kaguya**) exhibits extreme variations in solar illumination angles, spatial resolution (0.25 m to 80 m), viewing geometry, and spectral modalities.

**MoonFlower AI** is an offline-first, research-grade planetary image registration platform built around three foundational layers:
- **Layer 1 — Data & Controlled Benchmark:** 100% metadata-driven sensor adapters (OHRC, TMC-2, IIRS, LRO NAC, SELENE TC) and a procedural synthetic lunar benchmark with 7 controlled ground-truth test scenarios.
- **Layer 2 — Registration Engine:** Pre-matching radiometric and textural image quality checks, tiered 3-level overlap estimation, illumination-invariant representations (CLAHE, Sobel gradient, 2D phase congruency, IIRS PCA), Adaptive Hybrid correspondence engine, Occam's razor geometric model selection, Adaptive Spatial Match Balancer ($4 \times 4, 6 \times 6, 8 \times 8$), and genuine 2D parabolic sub-pixel peak optimization.
- **Layer 3 — Scientific Evidence:** Mathematically grounded evaluation metrics (Inlier Ratio, Transfer RMSE, Median Residual, Spatial Coverage, Spatial Uniformity), explainable quality score ($0 - 100$), rule-based failure diagnostics, side-by-side Benchmark Lab, and full export packaging (CSV, JSON, Markdown report).

---

## 2. Directory Structure

```
d:/SIH/
├── app.py                      # Mission Control Streamlit Scientific Console
├── cli.py                      # Standalone Headless Command-Line Interface
├── requirements.txt            # Python dependencies (Standardized on Python 3.11.x)
├── README.md                   # System documentation & quickstart
├── .env                        # Local runtime secrets (untracked)
├── .env.example                # Clean environment template
├── .gitignore                  # Git hygiene
│
├── config/
│   ├── __init__.py
│   └── settings.yaml           # Algorithm parameters, RANSAC thresholds, grid sizes
│
├── data/
│   ├── raw/                    # User uploaded raw GeoTIFF/TIFF/PDS files
│   ├── processed/              # Normalized, preprocessed intermediate caches
│   ├── samples/                # Built-in synthetic & reference lunar datasets with sidecars
│   ├── outputs/                # Registration outputs, GeoTIFFs, reports, metrics
│   └── weights/                # Local model weight caches (SuperPoint, LoFTR)
│
├── experiments/
│   ├── __init__.py
│   ├── tracker.py              # Experiment registry, JSON telemetry & CSV benchmark logger
│   ├── configs/                # Reproducible experiment configurations
│   ├── results/                # Cumulative experiments_log.csv registry
│   └── runs/                   # Detailed per-run JSON telemetry artifacts
│
├── pipeline/
│   ├── __init__.py
│   ├── registration_pipeline.py# Central orchestrator: result = pipeline.run(src, ref, cfg)
│   └── export.py               # Export engine: CSV, JSON, PNGs, and scientific report
│
├── preprocessing/
│   ├── __init__.py
│   ├── loader.py               # Standardized LunarImage loader (GeoTIFF, TIFF, PNG)
│   ├── normalization.py        # Radiometric stretch & histogram matching
│   ├── illumination.py         # CLAHE, Sobel gradient magnitude, Phase congruency
│   ├── hyperspectral.py        # IIRS 250-band cube handler, wavelength slicing & PCA
│   ├── tiling.py               # Memory-aware spatial chunking for gigapixel rasters
│   └── pyramid.py              # Coarse-to-fine Gaussian scale pyramid builder
│
├── metadata/
│   ├── __init__.py
│   ├── reader.py               # Metadata reader and tag extractor
│   ├── geometry.py             # Sun vector and GSD scale calculations
│   └── adapters.py             # Sensor adapters (OHRC, TMC-2, IIRS, LRO NAC, SELENE)
│
├── analysis/
│   ├── __init__.py
│   ├── image_quality.py        # Valid pixel %, contrast, texture entropy, dynamic range
│   ├── overlap.py              # Tiered Overlap (Level 1: Metadata, Level 2: Image-based, Level 3: Unknown)
│   └── failure_diagnostics.py  # Root-cause failure analysis & recommendation engine
│
├── features/
│   ├── __init__.py
│   ├── classical.py            # SIFT and ORB/AKAZE keypoint & descriptor extractors
│   └── learned.py              # SuperPoint / LoFTR abstraction with graceful offline fallback
│
├── matching/
│   ├── __init__.py
│   ├── result.py               # Standardized MatchResult dataclass contract
│   ├── classical.py            # FLANN matcher with Lowe's ratio test
│   ├── learned.py              # Learned/Dense matcher adapter
│   ├── hybrid.py               # Adaptive Hybrid Correspondence Engine (SIFT + Learned Fusion)
│   └── filtering.py            # Confidence thresholding & duplicate keypoint purging
│
├── registration/
│   ├── __init__.py
│   ├── transformation.py       # Translation, Similarity, Affine, Homography models
│   ├── model_selection.py      # Automated model selection based on residual analysis
│   ├── ransac.py               # Robust estimation via RANSAC and USAC_MAGSAC
│   ├── subpixel.py             # Local NCC & 2D parabolic quadratic peak optimization
│   ├── non_rigid.py            # Moran's I spatial residual strain & piecewise affine warp
│   └── warping.py              # Forward/inverse warping, difference heatmaps & overlays
│
├── spatial/
│   ├── __init__.py
│   ├── distribution.py         # Adaptive Spatial Match Balancer (ISRO Grid 4x4, 6x6, 8x8)
│   └── coverage.py             # 2D convex hull spatial coverage calculator
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # RMSE, Median Residual, Coverage, Uniformity
│   ├── quality_score.py        # Explainable Lunar Registration Quality Score (0–100)
│   └── benchmark.py            # Multi-method progressive comparative harness
│
├── visualization/
│   ├── __init__.py
│   ├── matches.py              # Correspondence ray rendering (Green inliers vs Red outliers)
│   └── diagnostics.py          # Residual histograms and error vector scatter
│
├── demo/
│   ├── __init__.py
│   ├── generator.py            # Procedural lunar DEM and photometric shading
│   └── scenarios.py            # 7 Controlled benchmark scenarios with model-specific GT
│
├── scripts/
│   └── generate_sample_datasets.py # Generates test lunar rasters and JSON sidecars
│
├── tests/
│   ├── test_foundation.py      # Foundation, config, hardware & path unit tests
│   ├── test_loader.py          # Data loader and adapter unit tests
│   ├── test_demo.py            # Synthetic benchmark and scenario GT tests
│   ├── test_analysis.py        # Image quality and tiered overlap tests
│   ├── test_preprocessing.py   # Illumination representations & PCA tests
│   ├── test_registration.py    # Feature extraction, RANSAC, model selection tests
│   ├── test_spatial.py         # Spatial coverage and match balancing tests
│   ├── test_subpixel.py        # Sub-pixel peak fitting & displacement tests
│   ├── test_metrics.py         # Scientific metrics and benchmark tests
│   ├── test_hybrid.py          # Adaptive hybrid engine, diagnostics & export tests
│   ├── test_pipeline.py        # End-to-end registration pipeline tests
│   ├── test_tracker.py         # Experiment registry and JSON/CSV logging tests
│   ├── test_integration.py     # Full-lifecycle multi-modal regression tests
│   └── test_cli.py             # Standalone CLI entrypoint tests
│
└── utils/
    ├── __init__.py
    ├── hardware.py             # CPU/CUDA/VRAM detection and memory management
    ├── logging.py              # Structured scientific console and file logging
    └── paths.py                # Safe platform-independent directory paths
```

---

## 3. Quickstart & Installation

```bash
# 1. Clone repository and navigate to workspace
cd d:/SIH

# 2. Install dependencies (Standardized on Python 3.11.x)
pip install -r requirements.txt

# 3. Initialize environment variables
copy .env.example .env

# 4. Generate built-in sample lunar datasets
python scripts/generate_sample_datasets.py

# 5. Run full automated test suite (53 tests across 14 suites)
python -m pytest tests/ -v

# 6. Launch Mission Control Console
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 4. Headless Scientific CLI (`cli.py`)

For automated ingest pipelines, batch processing, and headless server environments:

```bash
# Run a controlled synthetic scenario with full stdout summary
python cli.py --scenario scenario_1_small_scale --prep clahe --matcher hybrid

# Register local files and export full scientific package
python cli.py --source data/samples/sample_ohrc.tif \
              --reference data/samples/sample_tmc2.tif \
              --prep clahe --matcher sift \
              --export-dir data/outputs/my_run

# Execute progressive benchmark across 4 algorithmic stages
python cli.py --scenario scenario_5_illumination_shift --benchmark

# Output parseable JSON payload (ideal for scripting and CI/CD)
python cli.py --scenario scenario_1_small_scale --quiet --json
```

---

## 5. Key Contributions for the SIH Grand Finale

1. **Adaptive Hybrid Correspondence Engine:** Fuses classical SIFT with learned features based on geometric evidence, eliminating feature starvation on planar lunar maria.
2. **Adaptive Spatial Match Balancer:** Enforces uniform match point distribution across a $4 \times 4$, $6 \times 6$, or $8 \times 8$ grid, preventing crater rim over-clustering and satisfying ISRO's requirement for well-conditioned control points.
3. **Genuine Sub-Pixel Refinement:** Uses local patch NCC and 2D parabolic quadratic peak fitting to measure continuous sub-pixel adjustments without fabricated precision claims.
4. **Illumination-Robust Preprocessing:** Sobel gradient magnitude and 2D phase congruency representations maintain geometric consistency across $180^\circ$ solar shadow inversions.
5. **Tiered Overlap & Quality Intelligence:** Evaluates radiometric dynamic range, Shannon texture entropy, and 3-level overlap prior to matching.
6. **Scientific Benchmark Lab & Experiment Registry:** Progressive validation harness comparing SIFT Baseline vs SIFT+Balancer vs SIFT+Subpixel vs Full Pipeline, with immutable audit logging into JSON and CSV registries.
