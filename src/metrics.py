import numpy as np

def _prep(y, p):
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    assert y.shape == p.shape, f"shape mismatch {y.shape} vs {p.shape}"
    assert np.isfinite(p).all(), "predictions contain NaN or inf"
    return y, p

def mae(y, p):  
    y,p=_prep(y,p); return np.abs(y-p).mean()

def mse(y, p):   
    y,p=_prep(y,p); return ((y-p)**2).mean()

def rmse(y, p):  
    return np.sqrt(mse(y,p))

def wape(y, p):
    y,p = _prep(y,p)
    return np.abs(y-p).sum() / np.abs(y).sum()

def mape(y, p, eps=1e-8):
    y,p = _prep(y,p)
    return (np.abs(y-p) / np.maximum(np.abs(y), eps)).mean()

def smape(y, p, eps=1e-8):
    y,p = _prep(y,p)
    denom = np.maximum((np.abs(y)+np.abs(p))/2, eps)
    return (np.abs(y-p) / denom).mean()

def all_metrics(y, p):
    return {"wape": wape(y,p), "mae": mae(y,p), "rmse": rmse(y,p),
            "mse": mse(y,p), "mape": mape(y,p), "smape": smape(y,p)}