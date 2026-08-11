import torch
import torch.nn as nn
from ..scaling import RevIN


class MLPForecaster(nn.Module):
    """Two hidden layers, GELU activations. Tests whether nonlinearity helps."""
    def __init__(self, L, H, d_model=512, dropout=0.2, use_revin=True, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, H))
        self.revin = RevIN() if use_revin else None

    def forward(self, x_hist, x_fut=None, sidx=None):
        y = x_hist[..., -1]
        if self.revin:
            y = self.revin.norm(y)
        out = self.net(y)
        if self.revin:
            out = self.revin.denorm(out)
        return out