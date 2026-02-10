# KM Curve Estimator

This KM Curve Estimator solves the problem of translating KM curve images into workable digital data. The implementation takes inspiration from [this JHU preprint](https://www.biorxiv.org/content/10.1101/2025.09.15.676421v1.full) while making improvements to digitization, reproducibility, and lower cost per image.

Key Differentiators:
1. Zero-LLM digitization step -> reproducible and deterministic through DP tracing
2. Cost-per-image -> lower by using cheaper models, fewer calls, and offloading work to other processes
3. Synthetic test robustness -> Contains strong degradation to test robustness
4. High test accuracy -> Beats preprint KM-GPT by 2x on harder tests (0.0082 IAE)

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
git clone https://github.com/patrick-zx-wei/KM-curve-estimator
cd KM-curve-estimator
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### API Keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Required: `OPENAI_API_KEY` and `GEMINI_API_KEY`.

## Usage

### CLI

```bash
km-extract tests/fixtures/standard/case_001/input/graph.png -o output.json
```

### Reconstruction Benchmark

```bash
python tests/run_recon_bench.py                          # all 100 cases through reconstruction + validation
python tests/run_recon_bench.py case_093                 # single case
python tests/run_recon_bench.py --generate-cache         # re-digitize and cache information alongside reconstruction + validation
python tests/run_recon_bench.py --workers 3              # run in parallel
```

## How It Works

### 1. Preprocess

Upscales images to ~2000px, normalizes background colours, denoises, and then sharpens before grading image quality from 1-10.

### 2. Input validation

Uses Gemini to verify the image provided is a valid KM curve. Checks for clear axes, step-function curves, and tick marks. Non KM images are filtered at this step.

### 3. MMPU (Multi-Modality Processing Unit)

Extracts axes, tick labels, legend and risk table text using GPT-5 Mini with Gemini Flash for validation. Packages data into PlotMetadata. When extraction is poor, axis and risk tables are re-extracted. This step reduces costs against full GPT-5 or Gemini Pro models.

Axis: Applies geometric inference and checks ticks to reconstruct spacing. 

Risk table: Parses text tokens into time points to reconstruct at-risk counts.

### 4. Digitizer

Determines line type/colour of each curve, then scans plot region using probability maps. Each curve is path traced and censor marks are detected. Unlike KM-GPT, this step does not use LLMs to trace the curves.

First, a score map is constructed using this evidence grading to reward curves and suppress false positives.

| Signal | Weight | Method |
|---|---|---|
| Color | 0.52 | Lab color distance from legend samples |
| Ridge | 0.30 | Structure tensor eigenvalues |
| Edge | 0.20 | Canny edge detection |

| Penalty | Weight |
|---|---|
| Axis | 0.85 |
| Frame | 0.70 |
| Hough line | 0.68 |
| Text | 0.58 |

Then, dynamic programming is used to trace all curves simultaneously, column by column. Cost is calculated based on pixel score and smoothness. Curves crossing over each other are lightly penalized and collision detection is used to share candidate pixels when lines overlap.

Afterward, the curve is smoothed out by taking the median y for all columns mapping to the same x column. A running median filter clamps down on isolated spikes and trims off other artifacts.

Lastly, a series of checks makes sure the curve does not track axes or other curves. 

### 5. Reconstruction

Using the risk table, the pipeline follows Guyot's algorithm to find exact patient event/censor moments and reproduces detailed table. Intervals in which reconstruction is poor against the digitized are re-run to improve performance across specific intervals.

Next, the digitized and rerendered curves are plotted to measure overlap, and corrections are applied where they diverge.

Lastly, greedy event correction runs through all events, flipping between censor and event status and locking in flips if it improves the curve's MAE.

Without a risk table, cohort size is estimated and IPD is generated based on curve shape.

### 6. Validation

Reconstructs KM curve from previous step and compares against digitized curve. If MAE between the curves exceeds threshold, pipeline is re-run up to N times or until target performance is achieved.

## Design Decisions

1. Zero-LLM digitization:

Digitization is done using a probability map and column-based dynamic programming. This is reproducible and significantly cheaper per image since all processing is done without use of APIs.

2. Multi-arm DP:

All curves are traced simultaneously, meaning crossover and close cases are handled safely. 

3. Tiered model extraction:

Combining GPT-5 Mini and Gemini Flash 3.0 reduces API cost per image while maintaining sufficient levels of accuracy per image.

4. Multi-pass reconstruction:

Reconstruction and greedy flips help optimize events that may not have been caught in the first solution.

## Benchmark Results

100 cases, 0 failed, 207 arms total.

| Metric | Mean | Median |
|------------|------|------|
| Validation MAE (recon vs digitized) | 0.0042 | 0.0037 |
| MAE (recon vs GT) | 0.0105 | 0.0082 |
| IAE (recon vs GT) | 0.0105 | 0.0082 |
| RMSE (recon vs GT) | 0.0163 | 0.0116 |

Arms < 1% MAE: 137/207 (66%)
Arms < 2% MAE: 193/207 (93%)

**MAE** (Mean Absolute Error): Average of |S_true(t) − S_recon(t)| sampled at all step-change time points.

**IAE** (Integrated Absolute Error): ∫₀¹ |S_true(t) − S_recon(t)| dt, normalized by the follow-up time span. Measures total area between the true and reconstructed survival curves.

**RMSE** (Root Mean Square Error): √(mean of (S_true(t) − S_recon(t))²) sampled at all step-change time points. Penalizes larger deviations more heavily than MAE.


### Comparison to KM-GPT

KM-GPT reports a median IAE of 0.018 across 538 clean synthetic plots in ideal conditions. This project's median IAE of 0.0082 is less than half that, despite being evaluated on a deliberately difficult benchmark that includes JPEG compression, low resolution, noisy backgrounds, and other artifacts common in real publications.

In practice, KM plots scraped from PDFs or scanned from older papers are rarely perfect. Our benchmark is designed to reflect these real-world conditions, with 50% of test cases including at least one form of image degradation.

### Factors in Test Design

#### Characteristic probabilities:

The 100 test benchmark includes 50 pristine cases, 35 standard cases, and 15 legacy cases. 

| Modifier | Overall | Pristine | Standard | Legacy |
|---|---|---|---|---|
| Risk table | 100% | 100% | 100% | 100% |
| Censoring marks | 100% | 100% | 100% | 100% |
| JPEG artifacts | 45% | 0% | 100% | 67% |
| Grid lines | 44% | 32% | 63% | 40% |
| Annotations (p-values, HR) | 35% | 30% | 40% | 47% |
| Low resolution | 30% | 0% | 54% | 73% |
| Thick lines | 33% | ~30% | 37% | - |
| Truncated Y-axis | 12% | 10% | - | 33% |
| Noisy background | 9% | 0% | 0% | 60% |
| Thin lines | 7% | 3% | - | - |

#### Test case restrictions:

After upscaling, a minimum of 2.3px line thickness must be maintained for any given test case.

A maximum of two from {JPEGArtifacts, LowResolution, NoisyBackground} can be selected for any given test case.

A maximum of four from {JPEGArtifacts, LowResolution, NoisyBackground, ThinLines, TruncatedYAxis, Annotations, GridLines} can be selected for any given test case.

#### The curve patterns are distributed as follows:

| Pattern | Overall % |
|---|---|
| Parallel | 35.4% |
| Diverging | 26.0% |
| Converging | 27.0% |
| Crossover | 11.6% |

| Arms per case | Count |
|---|---|
| 1-arm | 9 |
| 2-arm | 75 |
| 3-arm | 16 |

## Outside of Project Scope

**Confidence interval bands**: KM plots with shaded CI regions around curves have not been tested. Adding confidence bands would require further tuning but should be achievable.

**No-risk-table reconstruction**: The pipeline supports estimated-mode reconstruction when no risk table is present, but all 100 benchmark cases include a risk table. Accuracy of the estimated path has not been formally benchmarked.

**Upwards Trajectory Graphs**: As outlined in project spec, not needed.

