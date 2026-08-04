"""Answer the questions that decide the model design."""
import json
import pandas as pd

RAW = "data/raw"

tr = pd.read_csv(f"{RAW}/train.csv", parse_dates=["timestamp"])
vi = pd.read_csv(f"{RAW}/validation_input.csv", parse_dates=["timestamp"])
fi = pd.read_csv(f"{RAW}/forecast_index_validation.csv", parse_dates=["timestamp"])
meta = json.load(open(f"{RAW}/metadata.json"))

def rule(t): print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)

# ---- 1. shapes and coverage ---------------------------------------
rule("1. SHAPES")
for name, d in [("train", tr), ("val_input", vi), ("fcst_index", fi)]:
    per = d.groupby("series_id").size()
    print(f"{name:11s} {d.shape}  series={d.series_id.nunique():3d}  "
          f"rows/series={per.min()}..{per.max()}")
    print(f"{'':11s} {d.timestamp.min()} -> {d.timestamp.max()}")

# ---- 2. THE question: are future covariates given? ----------------
rule("2. DO WE GET FEATURES FOR THE FORECAST WINDOW?")
fkeys = set(map(tuple, fi[["series_id", "timestamp"]].values))
vkeys = set(map(tuple, vi[["series_id", "timestamp"]].values))
covered = len(fkeys & vkeys)
print(f"forecast rows            : {len(fkeys)}")
print(f"of those, present in vi  : {covered}")
print(f"=> FUTURE COVARIATES     : {'YES' if covered == len(fkeys) else 'NO'}")
print(f"\nvalidation_input has target column: {'target' in vi.columns}")
if "target" in vi.columns:
    print(f"  ... non-null targets in it: {vi.target.notna().sum()}")

# ---- 3. the gap between train end and forecast start --------------
rule("3. TIMELINE")
gap = (fi.timestamp.min() - tr.timestamp.max()).total_seconds() / 3600
print(f"train ends        {tr.timestamp.max()}")
print(f"forecast starts   {fi.timestamp.min()}")
print(f"gap               {gap:.0f} hours")
print(f"forecast length   {fi.groupby('series_id').size().iloc[0]} hours")

# ---- 4. missing values --------------------------------------------
rule("4. MISSING VALUES (train)")
miss = tr.isna().mean().sort_values(ascending=False)
print(miss[miss > 0].round(4).to_string() or "none")

# ---- 5. which features never change inside a series? --------------
rule("5. STATIC vs TIME-VARYING")
feats = [c for c in tr.columns if c not in ("series_id", "timestamp", "target")]
nun = tr.groupby("series_id")[feats].nunique().max()
for c in feats:
    print(f"  {'STATIC     ' if nun[c] == 1 else 'time-varying'} {c}")

# ---- 6. target ------------------------------------------------------
rule("6. TARGET")
print(tr.target.describe().round(3).to_string())
pm = tr.groupby("series_id").target.mean()
print(f"\nper-series mean: min={pm.min():.2f}  max={pm.max():.2f}  "
      f"ratio={pm.max()/pm.min():.1f}x")
print(f"target NaNs    : {tr.target.isna().sum()}")

rule("metadata.json")
print(json.dumps(meta, indent=2)[:1200])