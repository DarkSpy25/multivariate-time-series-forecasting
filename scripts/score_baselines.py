import pandas as pd
from src.baselines import make_all_baselines
from src.metrics import all_metrics
from src.data import fold_bounds

tr = pd.read_csv("data/raw/train.csv", parse_dates=["timestamp"])
hours = sorted(tr.timestamp.unique())          # 4320 unique hours

rows = []
for fold in range(3):
    tr_end, s0, s1 = fold_bounds(fold)
    hist   = tr[tr.timestamp <  hours[tr_end]]
    # truth  = tr[(tr.timestamp >= hours[s0]) & (tr.timestamp < hours[s1])]
    if s1 == len(hours):
        truth = tr[tr.timestamp >= hours[s0]]
    else:
        truth = tr[
            (tr.timestamp >= hours[s0]) &
            (tr.timestamp < hours[s1])
        ]
    index  = truth[["series_id","timestamp"]].copy()

    for name, pred in make_all_baselines(hist, index).items():
        m = truth.merge(pred, on=["series_id","timestamp"], how="left")
        assert m.prediction.notna().all(), f"{name}: missing rows"
        rows.append({"model": name, "fold": fold, **all_metrics(m.target, m.prediction)})

df = pd.DataFrame(rows)
print(df.groupby("model")[["wape","mae","rmse","smape"]].agg(["mean","std"]).round(4))
df.to_csv("experiments/baselines.csv", index=False)