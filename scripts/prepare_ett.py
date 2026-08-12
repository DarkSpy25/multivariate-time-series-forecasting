import pandas as pd, numpy as np, os

df = pd.read_csv("data/ett/ETTh1.csv", parse_dates=["date"])
df = df.rename(columns={"date": "timestamp", "OT": "target"})
df["series_id"] = "ett_h1"                      # a single "unit"

# calendar features, the same ones the main dataset ships with
h = df["timestamp"].dt.hour
d = df["timestamp"].dt.dayofweek
df["hour_sin"] = np.sin(2*np.pi*h/24); df["hour_cos"] = np.cos(2*np.pi*h/24)
df["dow_sin"]  = np.sin(2*np.pi*d/7);  df["dow_cos"]  = np.cos(2*np.pi*d/7)
df["is_weekend"] = (d >= 5).astype("float32")

os.makedirs("data/ett", exist_ok=True)
df.to_csv("data/ett/long.csv", index=False)
print(df.shape, "-> wrote data/ett/long.csv")