import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

L = 512 
H = 336            # forecast horizon, fixed by the task
N_TRAIN_HOURS = 4320
CAL     = ["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","trend"]
STATIC  = ["nominal_capacity","zone_sin","zone_cos"]

def fold_bounds(fold: int, total: int = N_TRAIN_HOURS, h: int = H):
    """Return (train_end, score_start, score_end) as hour indices."""
    score_end   = total - fold * h
    score_start = score_end - h
    train_end   = score_start   # training must not touch [score_start, score_end)
    return train_end, score_start, score_end


def load_cube(path, use_trend=True, fill="ffill", add_mask=False):
    """long CSV -> (X, Y, series_ids, hours, feat_names, static)

    X : (S, T, F) float32   dynamic features, scaled later
    Y : (S, T)    float32   target
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["series_id","timestamp"]).reset_index(drop=True)

    series = sorted(df.series_id.unique())          # FIXED order, save it
    hours  = np.sort(df.timestamp.unique())
    S, T   = len(series), len(hours)

    feats = [c for c in df.columns
             if c not in ("series_id","timestamp","target") + tuple(STATIC)]
    if not use_trend: feats = [c for c in feats if c != "trend"]

    gappy = [c for c in feats if df[c].isna().any()]
    if add_mask:
        for c in gappy:
            df[c + "_isna"] = df[c].isna().astype("float32")
        feats += [c + "_isna" for c in gappy]

    # fill INSIDE each series -- never across the boundary
    if fill == "ffill":
        df[gappy] = df.groupby("series_id")[gappy].ffill()
        df[gappy] = df.groupby("series_id")[gappy].bfill()
    elif fill == "mean":
        df[gappy] = df[gappy].fillna(df[gappy].mean())
    df[gappy] = df[gappy].fillna(0.0)

    X = df[feats].to_numpy("float32").reshape(S, T, len(feats))
    Y = df["target"].to_numpy("float32").reshape(S, T)
    have = [c for c in STATIC if c in df.columns]
    if have:
        St = df.groupby("series_id")[have].first().loc[series].to_numpy("float32")
    else:
        St = np.zeros((S, 0), "float32")  # ETT has no static cols; St is unused downstream

    assert np.isfinite(X).all() and np.isfinite(Y).all()
    return X, Y, series, hours, feats, St



class WindowDataset(Dataset):
    """Every legal (series, start-hour) pair inside [0, train_end)."""
    def __init__(self, X, Y, L, H, train_end, gap=0):
        self.X, self.Y, self.L, self.H, self.gap = X, Y, L, H, gap
        S = X.shape[0]
        n_starts = train_end - L - gap - H + 1
        assert n_starts > 0, f"L={L}+gap={gap}+H={H} exceeds {train_end} hours"
        self.index = [(s, t) for s in range(S) for t in range(n_starts)]
        self.train_end = train_end

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        s, t = self.index[i]
        L, H, g = self.L, self.H, self.gap
        assert t + L + g + H <= self.train_end        # no peeking
        f0 = t + L + g                                 # forecast origin
        x_hist = self.X[s, t : t+L]                    # (L, F)
        y_hist = self.Y[s, t : t+L, None]              # (L, 1)
        x_fut  = self.X[s, f0 : f0+H]                  # (H, F) known covariates
        y_fut  = self.Y[s, f0 : f0+H]                  # (H,)   the answer
        return (torch.from_numpy(np.concatenate([x_hist, y_hist], 1)),
                torch.from_numpy(x_fut),
                torch.from_numpy(y_fut),
                s)

def make_loader(ds, bs=64, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=0, drop_last=shuffle)


def validation_batch(Xs, Y, series, L, train_end, s0, s1, gap=0):
    """One window per series, history ending gap hours before the scored block."""
    hist_end = train_end - gap
    assert hist_end - L >= 0, f"L={L}+gap={gap} exceeds {train_end} hours"
    xh = np.stack([np.concatenate(
        [Xs[s, hist_end-L:hist_end], Y[s, hist_end-L:hist_end, None]], 1)
        for s in range(len(series))]).astype("float32")
    xf = Xs[:, s0:s1].astype("float32")
    return torch.from_numpy(xh), torch.from_numpy(xf), Y[:, s0:s1]

def prepare_inference_cube(inp_path, fi, features, series_order, scaler, L, H,
                           fallback_x=None, fallback_y=None, fallback_end=None):
    """Inference tensors. The graded input is covariates-only (no target),
    so the past-target history comes from the baked checkpoint tail."""
    df = pd.read_csv(inp_path, parse_dates=["timestamp"])
    df = df.sort_values(["series_id", "timestamp"])
    S = len(series_order)

    # future covariates -> (S, H, F) in the saved feature order, then scaled
    fut = np.sort(pd.to_datetime(fi.timestamp).unique())[-H:]
    Xf = np.zeros((S, H, len(features)), "float32")
    for i, sid in enumerate(series_order):
        d = df[df.series_id == sid].set_index("timestamp").reindex(fut)
        d[features] = d[features].ffill().bfill()
        Xf[i] = d[features].to_numpy("float32")
    Xf = np.nan_to_num(Xf, nan=0.0)
    Xf = scaler.transform(Xf).astype("float32")

    # history -> (S, L, F+1): baked scaled covariates + baked raw target
    assert fallback_x is not None, ("checkpoint has no baked history; "
        "retrain after step 8.5.2")
    hx = np.asarray(fallback_x, "float32")             # (S, L, F) scaled
    hy = np.asarray(fallback_y, "float32")[..., None]  # (S, L, 1) raw
    Xh = np.concatenate([hx, hy], 2).astype("float32")
    return Xh, Xf, list(series_order)

if __name__ == "__main__":
    X, Y, series, hours, feats, St = load_cube("data/raw/train.csv")
    ds = WindowDataset(X, Y, L=512, H=336, train_end=3984, gap=0)
    print(len(ds))                    # ~301,000
    a,b,c,d = ds[0]
    print(a.shape, b.shape, c.shape)  # (512,23) (336,22) (336,)