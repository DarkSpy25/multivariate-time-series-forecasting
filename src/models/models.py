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
    """Dispatcher: turn --model name into the right class."""
    if args.model == "linear":
        return LinearForecaster(L=L, H=H, n_feat=n_feat,
                             use_revin=bool(args.revin),
                             use_covariates=bool(args.covariates))
    if args.model == "mlp":
        return MLPForecaster(L=L, H=H, use_revin=bool(args.revin))
    if args.model == "lstm":
        return LSTMForecaster(L=L, H=H, n_feat=n_feat, use_revin=bool(args.revin))
        return LinearForecaster(L=L, H=H, use_revin=bool(args.revin))
    if args.model == "patchtst":
        return PatchTST(
            L=L,
            H=H,
            n_feat=n_feat,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            layers=args.layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            use_revin=bool(args.revin),
            channel_mixing=bool(args.channel_mixing),
            target_only=bool(args.target_only),
        )
    raise ValueError(f"Unknown model: {args.model}")


def build_model_from_config(cfg, n_feat, n_series, L, H):
    """Rebuild a model from a saved config dict (used by predict.py later)."""
    if cfg["model"] == "linear":
        return LinearForecaster(L=L, H=H, use_revin=bool(cfg.get("revin", 1)))
    if cfg["model"] == "patchtst":
        return PatchTST(
            L=L,
            H=H,
            n_feat=cfg.get("n_feat", 1),
            patch_len=cfg.get("patch_len", 16),
            stride=cfg.get("stride", 8),
            d_model=cfg.get("d_model", 128),
            layers=cfg.get("layers", 3),
            n_heads=cfg.get("n_heads", 8),
            d_ff=cfg.get("d_ff", 256),
            dropout=cfg.get("dropout", 0.2),
            use_revin=bool(cfg.get("revin", 1)),
            channel_mixing=bool(cfg.get("channel_mixing", False)),
            target_only=bool(cfg.get("target_only", False)),
        )
    raise ValueError(f"Unknown model: {cfg['model']}")

class PatchTST(nn.Module):
    def __init__(self, L, H, n_feat, patch_len=16, stride=8,
                 d_model=128, layers=3, n_heads=8, d_ff=256,
                 dropout=0.2, use_revin=True, channel_mixing=False,
                 target_only=False, **_):
        super().__init__()
        self.pl, self.st = patch_len, stride
        self.n_patch = (L - patch_len) // stride + 1
        self.C = 1 if target_only else n_feat + 1
        self.mixing, self.target_only = channel_mixing, target_only

        # 1. each patch -> a d_model vector
        self.embed = nn.Linear(patch_len, d_model)
        # 2. learned position per patch
        self.pos   = nn.Parameter(torch.randn(1, self.n_patch, d_model) * 0.02)
        self.drop  = nn.Dropout(dropout)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(enc, num_layers=layers)

        # 3. flatten all patch vectors -> H predictions
        head_in = self.n_patch * d_model * (self.C if channel_mixing else 1)
        self.head  = nn.Linear(head_in, H)
        self.revin = RevIN() if use_revin else None

    def _patchify(self, z):                       # z (N, L)
        return z.unfold(dimension=1, size=self.pl, step=self.st)  # (N, P, pl)

    def forward(self, x_hist, x_fut=None, sidx=None):
        B, L, _ = x_hist.shape
        x = x_hist[..., -1:] if self.target_only else x_hist
        if self.revin:
            tgt = self.revin.norm(x_hist[..., -1])              # (B, L)
            x = torch.cat([x[..., :-1], tgt.unsqueeze(-1)], -1)

        z = x.permute(0, 2, 1).reshape(B * self.C, L)      # (B*C, L)
        p = self.embed(self._patchify(z)) + self.pos           # (B*C, P, d)
        h = self.backbone(self.drop(p))                        # (B*C, P, d)

        if self.mixing:
            h = h.reshape(B, self.C * self.n_patch * h.size(-1))
            out = self.head(h)                                 # (B, H)
        else:
            out = self.head(h.reshape(B * self.C, -1))         # (B*C, H)
            out = out.reshape(B, self.C, -1)[:, -1]           # target channel

        if self.revin: out = self.revin.denorm(out)
        return out