import torch.nn as nn


class CovariateHead(nn.Module):
    """Squeeze (B, H, F) known-future covariates into an additive (B, H)."""
    def __init__(self, n_feat, H, d=64, dropout=0.1):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(n_feat, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.mix = nn.Linear(H, H)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x_fut):
        return self.mix(self.point(x_fut).squeeze(-1))