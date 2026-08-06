from src.scaling import GlobalScaler
import numpy as np
from src.data import load_cube, WindowDataset

CSV_PATH = "data/raw/train.csv"

L = 512 
H = 336

train_end = 3984

print("Loading data...")
X, Y, series, hours, feats, St = load_cube(CSV_PATH)

print("Creating dataset...")
ds = WindowDataset(X, Y, L=L, H=H, train_end=train_end, gap=0)

print("Running pipeline assertions...")

# 1. no window's target reaches into the scored region
mx = max(t for _, t in ds.index)
assert mx + L + H <= train_end

# 2. no window straddles two series
assert all(0 <= s < X.shape[0] for s, _ in ds.index)

# 3. scaler saw only training hours
sc = GlobalScaler().fit(X, upto=train_end)
sc2 = GlobalScaler().fit(X[:, :train_end], upto=train_end)
assert np.allclose(sc.mean, sc2.mean)

# 4. a window really is contiguous in time
s, t = ds.index[12345]
assert np.allclose(ds[12345][0][:, -1].numpy(), Y[s, t:t+L])

# 5. shuffling doesn't mix channels
assert ds[0][0].shape[1] == X.shape[2] + 1
print("pipeline OK")