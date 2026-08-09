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