# dlam-ts-38

## Data contract
- 96 series, unit_000..unit_095
- train.csv: 4320 h/series, 2023-01-01 00:00 -> 2023-06-29 23:00, hourly, no gaps
- 22 feature columns + target; target has no NaNs
- static per series: nominal_capacity, zone_sin, zone_cos
- calendar: hour_sin/cos, dow_sin/cos, is_weekend, trend
- NaNs in: 
- future covariates supplied: YES / NO  <- from step 1.3
- horizon: 336 h; submission rows = 96 x 336 = 32,256
## Model design decisions
- Series identity: none used (relying on RevIN normalization + covariate pathway).
  Verified all validation_input.csv series exist in train.csv (empty set difference),
  so a learned embedding was technically viable — skipped for now given time,
  documented as a future ablation.
## Inference / Submission (Phase 9)

The submission archive contains: `predict.py`, `requirements.txt`,
`checkpoint.pt`, `src/`, `README.md` (this file). No training data,
virtual environments, notebooks, or experiment logs are included.

### Install

```
pip install -r requirements.txt
```

### Run

```
python predict.py --input_dir <path-to-input-dir> --output_file <path-to-output-file>/predictions.csv --checkpoint checkpoint.pt
```

- `--input_dir` must contain either `validation_input.csv` +
  `forecast_index_validation.csv`, or the graded `test_input.csv` +
  `forecast_index_test.csv` (predict.py auto-detects which).
- `--output_file` is the full path (including filename) to write the
  prediction CSV to; the parent directory is created if it doesn't exist.
- `--checkpoint` defaults to `checkpoint.pt` next to the script.

### Expected input / output schema

- Input covariates CSV: `series_id, timestamp, <22 feature columns>` (no
  target column -- the model's own training-history tail, baked into the
  checkpoint, supplies past target values).
- `forecast_index_*.csv`: `series_id, timestamp` -- the exact rows to
  predict.
- Output `predictions.csv`: `series_id, timestamp, prediction`, one row per
  (series, hour) pair -- 96 series x 336 hours = 32,256 rows + header.

### Hardware

Runs on CPU or Apple Silicon (MPS) automatically; no GPU required. Full
inference over all 96 series completes in well under a minute on a laptop.
