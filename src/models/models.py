import torch
import torch.nn as nn
from ..scaling import RevIN
from .mlp import MLPForecaster
from .lstm import LSTMForecaster
from .covariate_head import CovariateHead


class LinearForecaster(nn.Module):
    """Map L past target values straight to H future ones."""
    def __init__(self, L, H, n_feat=0, use_revin=True, use_covariates=False, **_):
        super().__init__()
        self.proj = nn.Linear(L, H)
        self.revin = RevIN() if use_revin else None
        self.cov = CovariateHead(n_feat, H) if use_covariates else None

    def forward(self, x_hist, x_fut=None, sidx=None):
        y = x_hist[..., -1]
        if self.revin:
            y = self.revin.norm(y)
        out = self.proj(y)
        if self.revin:
            out = self.revin.denorm(out)
        if self.cov is not None and x_fut is not None:
            out = out + self.cov(x_fut)
        return out


def build_model(args, n_feat, n_series, L, H):
    if args.model == "linear":
        return LinearForecaster(L=L, H=H, n_feat=n_feat,
                             use_revin=bool(args.revin),
                             use_covariates=bool(args.covariates))
    if args.model == "mlp":
        return MLPForecaster(L=L, H=H, use_revin=bool(args.revin))
    if args.model == "lstm":
        return LSTMForecaster(L=L, H=H, n_feat=n_feat, use_revin=bool(args.revin))
    raise ValueError(f"Unknown model: {args.model}")


def build_model_from_config(cfg, n_feat, n_series, L, H):
    """Rebuild a model from a saved config dict (used by predict.py later)."""
    if cfg["model"] == "linear":
        return LinearForecaster(L=L, H=H, use_revin=bool(cfg.get("revin", 1)))
    raise ValueError(f"Unknown model: {cfg['model']}")