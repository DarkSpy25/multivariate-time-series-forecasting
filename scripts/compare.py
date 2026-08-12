import pandas as pd
F = ["ts","run_id","model","fold","seed","lookback","loss","revin","covariates",
     "patch_len","d_model","layers","lr","epochs","wape","mae","rmse","mse",
     "mape","smape","minutes","notes"]
df = pd.read_csv("experiments/runs.csv", names=F, header=None)
df = df[df.notes.isin(["graded", "ett"])]          # just this comparison's runs
tbl = df.groupby(["notes", "model"]).wape.agg(["mean", "std", "count"]).round(4)
print(tbl.sort_values(["notes", "mean"]))          # lower WAPE = better