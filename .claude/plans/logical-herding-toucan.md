# Reorganize Test Fixtures + Remove Upward Curves

## Context
1. The problem spec says KM curves are monotonically decreasing — remove upward (incidence) test cases
2. `input/` should only contain `graph.png` (the sole pipeline input) to make zero data leakage obvious. Everything else moves to `ground_truth/` or `output/`.

## New directory layout
```
case_NNN/
  input/        graph.png
  ground_truth/ metadata.json, risk_table_data.csv, ground_truth.csv,
                raw_survival_data.csv, hard_points.json
  output/       digitized_cache.json, results.json, overlay_results.png
```

## Changes

### 1. Move metadata.json + risk_table_data.csv from input/ to ground_truth/
- Physically move files for all 100 cases
- `tests/synthetic/ground_truth.py` — `save_test_case()`: write metadata.json and risk_table_data.csv to `gt_dir` instead of `input_dir`
- `tests/synthetic/ground_truth.py` — `load_test_case()`: read metadata.json and risk_table_data.csv from `gt_dir`
- `tests/synthetic/ground_truth.py` — update module docstring

### 2. Update benchmark paths
- `tests/run_recon_bench.py` — read metadata.json from `ground_truth/`, risk_table_data.csv from `ground_truth/`
- `tests/run_recon_bench.py` — read/write digitized_cache.json from `output/` (create output/ dir if needed)

### 3. Update runner paths
- `tests/synthetic/runner.py` — `run_case()`: read metadata from `ground_truth/`, write results.json and overlay_results.png to `output/`
- `tests/synthetic/runner.py` — `_write_overlay_artifact()`: read graph.png from `input/` (already correct), write overlay to `output/`

### 4. Remove upward direction from presets
- `tests/synthetic/presets.py` — change direction_schedule to `["downward"] * n_total`
- Also remove `CurveDirection` import and `CurveDirectionLabel` type alias if no longer needed (check usage first)

### 5. Regenerate 15 upward cases
- Run `generate_standard()` to regenerate all cases with downward direction
- Only the 15 formerly-upward cases will produce different images
- Delete all existing `digitized_cache.json` files (they'll need re-digitization for changed cases)

## Files to modify
- `tests/synthetic/ground_truth.py` — save/load paths
- `tests/synthetic/runner.py` — read/write paths
- `tests/run_recon_bench.py` — read paths + cache path
- `tests/synthetic/presets.py` — direction schedule
- `.gitignore` — already covers digitized_cache.json

## Verification
- `grep -r '"curve_direction": "upward"' tests/fixtures/standard/` → zero matches
- `ls tests/fixtures/standard/case_001/input/` → only graph.png
- `ls tests/fixtures/standard/case_001/ground_truth/` → metadata.json, risk_table_data.csv, ground_truth.csv, raw_survival_data.csv, hard_points.json
- `python tests/run_recon_bench.py` → runs successfully (user verifies at home)
