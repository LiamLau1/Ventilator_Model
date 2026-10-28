"""Baseline LSTM with an information ladder: plant-causal / filter-causal / bidirectional."""
import torch
from torch import nn

from ..features import n_features

P_SCALE, P_OFFSET = 10.0, 8.0  # fixed output scaling: pressure = P_OFFSET + P_SCALE * head(h)


class LSTMBaseline(nn.Module):
    """variant: 'plant' (input uses u[k-1]), 'filter' (u[k], unidirectional), 'bi' (bidirectional).
    Input features come from vent.features.build_features(d, variant, track)."""

    def __init__(self, variant="plant", track="B", hidden=256, layers=2):
        super().__init__()
        assert variant in ("plant", "filter", "bi") and track in ("A", "B")
        self.variant, self.track, self.hidden, self.layers = variant, track, hidden, layers
        self.dirs = 2 if variant == "bi" else 1
        self.n_in = n_features(track)
        self.lstm = nn.LSTM(self.n_in, hidden, layers, batch_first=True, bidirectional=variant == "bi")
        self.head = nn.Linear(hidden * self.dirs, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return P_OFFSET + P_SCALE * self.head(h).squeeze(-1)

    def step(self, x_t, state=None):
        """One streaming step (unidirectional variants). x_t: (B, F). Returns (pressure (B,), state)."""
        if self.dirs == 2:
            raise NotImplementedError("bidirectional model cannot stream")
        h, state = self.lstm(x_t.unsqueeze(1), state)
        return P_OFFSET + P_SCALE * self.head(h[:, 0]).squeeze(-1), state

    def flops_per_breath(self, T=80):
        """Analytic: 2*4*(in+h)*h per layer-step (x2 if bidirectional) + head; elementwise ops ignored."""
        f, d_in = 0, self.n_in
        for _ in range(self.layers):
            f += 2 * 4 * (d_in + self.hidden) * self.hidden * self.dirs
            d_in = self.hidden * self.dirs
        f += 2 * d_in
        return f * T

    def config(self):
        return dict(variant=self.variant, track=self.track, hidden=self.hidden, layers=self.layers)
