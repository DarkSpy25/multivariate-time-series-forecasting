import torch
import torch.nn as nn
from ..scaling import RevIN


class LinearForecaster(nn.Module):
    """Map L past target values straight to H future ones."""
    def __init__(self, L, H, use_revin=True, **_):
        super().__init__()
        self.proj = nn.Linear(L, H)
        self.revin = RevIN() if use_revin else None

    def forward(self, x_hist, x_fut=None, sidx=None):
        y = x_hist[..., -1]
        if self.revin:
            y = self.revin.norm(y)
        out = self.proj(y)
        if self.revin:
            out = self.revin.denorm(out)
        return out


def build_model(args, n_feat, n_series, L, H):
    if args.model == "linear":
        return LinearForecaster(L=L, H=H, use_revin=bool(args.revin))
    raise ValueError(f"Unknown model: {args.model}")


def build_model_from_config(cfg, n_feat, n_series, L, H):
    if cfg["model"] == "linear":
        return LinearForecaster(L=L, H=H, use_revin=bool(cfg.get("revin", 1)))
    raise ValueError(f"Unknown model: {cfg['model']}")