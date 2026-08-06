import numpy as np, torch, torch.nn as nn

class GlobalScaler:
    """Per-feature standardisation. Fit on training hours only."""
    def fit(self, X, upto):                    # X (S,T,F)
        flat = X[:, :upto, :].reshape(-1, X.shape[2])
        self.mean = flat.mean(0)
        self.std  = flat.std(0) + 1e-6
        return self
    def transform(self, X):  return (X - self.mean) / self.std
    def state(self):        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
    @staticmethod
    def from_state(d):
        s = GlobalScaler(); s.mean = np.array(d["mean"], "float32")
        s.std = np.array(d["std"], "float32"); return s


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., ICLR 2022).

    Normalise each window by ITS OWN mean/std, predict, then undo it.
    Handles both scale differences between units and drift over time.
    """
    def __init__(self, eps=1e-5):
        super().__init__(); self.eps = eps; self.mu = None; self.sd = None

    def norm(self, x):                      # x (B, L) or (B, L, C)
        dim = 1
        self.mu = x.mean(dim, keepdim=True)
        self.sd = x.std(dim, keepdim=True) + self.eps
        return (x - self.mu) / self.sd

    def denorm(self, y):                    # y (B, H)
        mu, sd = self.mu, self.sd
        if mu.dim() == 3: mu, sd = mu.squeeze(-1), sd.squeeze(-1)
        return y * sd + mu