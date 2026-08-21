import pandas as pd
F = ["ts","run_id","model","fold","seed","lookback","loss","revin","covariates","patch_len","d_model","layers","lr","epochs","wape","mae","rmse","mse","mape","smape","minutes","notes"]
r = pd.read_csv("experiments/runs.csv", names=F, header=None)   # runs.csv has no header row

t1 = (r[r.notes.isin(["graded", "gradedcov"])]     # model comparison, already run
      .groupby(["notes", "model"])[["wape","mae","rmse","smape"]]
      .agg(["mean","std"]).round(4))

fin  = r[r.notes == "final"]
noise = fin.groupby("fold").wape.std().mean()
print(f"seed noise (mean std within fold): {noise:.4f}")

base = fin[fin.fold == 0].wape.mean()   # ablations run at fold 0, so compare to BEST at fold 0 (not the 3-fold mean)
t2 = r[r.notes.str.startswith("abl:", na=False)].copy()
t2["delta_%"]      = (100 * (t2.wape - base) / base).round(1)
t2["above_noise"]  = (t2.wape - base).abs() > 2 * noise

print(t1.to_latex())
print(t2[["notes","wape","delta_%","above_noise"]].to_latex(index=False))