"""Tiny checks: metrics on hand-built arrays, plant-causality, step() == forward.

Run:  conda run -n datasci_m1 python tests/test_basic.py   (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from vent.features import build_features
from vent.metrics import evaluate, insp_mask
from vent.models.lstm import LSTMBaseline


def test_metrics():
    uo = np.array([[0, 0, 0, 1] + [1] * 76, [0, 0, 1, 1] + [1] * 76])
    p = np.zeros((2, 80))
    pred = np.zeros((2, 80))
    pred[0, :4] = [1, 2, 3, 100]      # step 3 is expiratory: must be ignored
    pred[1, :4] = [4, 0, 0, 100]
    R, C, pi = np.array([5, 20]), np.array([10, 10]), np.array([True, False])
    assert insp_mask(uo, "A")[:, 0].sum() == 0 and insp_mask(uo, "B")[:, 0].sum() == 2
    b = evaluate(pred, p, uo, R, C, pi, "B")          # errors on scored steps: 1,2,3,4,0 -> 10/5
    assert abs(b["mae"] - 2.0) < 1e-12 and b["n_steps"] == 5
    assert abs(b["pi"]["pi"] - 2.0) < 1e-12 and abs(b["pi"]["non_pi"] - 2.0) < 1e-12
    assert abs(b["per_rc"]["R5_C10"] - 2.0) < 1e-12 and b["per_rc"]["R20_C10"] == 2.0 and len(b["per_rc"]) == 2
    assert b["per_step"][0] == 2.5 and b["per_step"][1] == 1.0 and b["per_step"][2] == 3.0 and b["per_step"][3] is None
    a = evaluate(pred, p, uo, R, C, pi, "A")          # step 0 dropped: errors 2,3,0 -> 5/3
    assert abs(a["mae"] - 5 / 3) < 1e-12 and a["per_step"][0] is None and a["n_steps"] == 3


def _data(n=6, seed=0):
    r = np.random.default_rng(seed)
    t = np.cumsum(np.full((n, 80), 0.0335), 1) - 0.0335
    return dict(t=t, u=r.uniform(0, 100, (n, 80)), uo=(np.arange(80) >= 30).astype(np.int8)[None].repeat(n, 0),
                R=r.choice([5, 20, 50], n), C=r.choice([10, 20, 50], n), p=r.normal(7, 1, (n, 80)))


def _fwd(model, d):
    with torch.no_grad():
        return model(torch.from_numpy(build_features(d, model.variant, model.track))).numpy()


def test_causality():
    d, k = _data(), 17
    d2 = {**d, "u": d["u"].copy(), "uo": d["uo"].copy()}
    d2["u"][:, k] += 37.0
    d2["uo"][:, k] = 1 - d2["uo"][:, k]
    for track in "AB":
        for variant, first_changed in (("plant", k + 1), ("filter", k), ("bi", 0)):
            torch.manual_seed(0)
            m = LSTMBaseline(variant, track, hidden=16, layers=2).eval()
            a, b = _fwd(m, d), _fwd(m, d2)
            same = np.abs(a - b).max(0) < 1e-7        # per step: unchanged by the perturbation of step k
            assert same[:first_changed].all(), (variant, track)
            assert not same[first_changed], (variant, track)   # and the first allowed step does react


def test_step_matches_forward():
    d = _data()
    for variant in ("plant", "filter"):
        m = LSTMBaseline(variant, "A", hidden=16, layers=2).eval()
        x = torch.from_numpy(build_features(d, variant, "A"))
        with torch.no_grad():
            full, state, outs = m(x), None, []
            for k in range(80):
                o, state = m.step(x[:, k], state)
                outs.append(o)
        assert torch.allclose(full, torch.stack(outs, 1), atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
