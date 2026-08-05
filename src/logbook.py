
import csv, json, time, pathlib

FIELDS = ["ts","run_id","model","fold","seed","lookback","loss","revin",
          "covariates","patch_len","d_model","layers","lr","epochs",
          "wape","mae","rmse","mse","mape","smape","minutes","notes"]

def log_run(path="experiments/runs.csv", **kw):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    row = {k: kw.get(k, "") for k in FIELDS}
    row["ts"] = time.strftime("%Y-%m-%d %H:%M")
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)