import torch
import torch.nn as nn
from ..scaling import RevIN


class LSTMForecaster(nn.Module):
    """Reads the window one hour at a time; final hidden state -> H predictions."""
    def __init__(self, L, H, n_feat, d_model=128, layers=2,
                 dropout=0.2, use_revin=True, **_):
        super().__init__()
        self.lstm = nn.LSTM(n_feat + 1, d_model, num_layers=layers,
                             batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(d_model, H)
        self.revin = RevIN() if use_revin else None

    def forward(self, x_hist, x_fut=None, sidx=None):
        x = x_hist
        if self.revin:
            y = self.revin.norm(x[..., -1])
            x = torch.cat([x[..., :-1], y.unsqueeze(-1)], -1)
        h, _ = self.lstm(x)
        out = self.head(h[:, -1])
        if self.revin:
            out = self.revin.denorm(out)
        return out