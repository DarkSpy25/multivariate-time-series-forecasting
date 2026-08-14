import torch
import torch.nn as nn
from ..scaling import RevIN
from .covariate_head import CovariateHead


class MLPForecaster(nn.Module):
    """Two hidden layers, GELU activations. Tests whether nonlinearity helps."""
    def __init__(self, L, H, n_feat=0, d_model=512, dropout=0.2,
                 use_revin=True, use_covariates=False, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, H))
        self.revin = RevIN() if use_revin else None
        self.cov = CovariateHead(n_feat, H) if use_covariates else None

    def forward(self, x_hist, x_fut=None, sidx=None):
        y = x_hist[..., -1]
        if self.revin:
            y = self.revin.norm(y)
        out = self.net(y)
        if self.revin:
            out = self.revin.denorm(out)
        if self.cov is not None and x_fut is not None:
            out = out + self.cov(x_fut)
        return out