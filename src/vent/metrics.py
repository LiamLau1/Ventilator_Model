"""Inspiratory MAE with breakdowns. Same protocol for every model."""
import json
from pathlib import Path

import numpy as np

N_STEPS = 32  # inspiration lasts 25-32 steps


def insp_mask(uo, track):
    """Scored steps: u_out == 0; on track A step 0 is also excluded (p[0] is an input)."""
    m = np.asarray(uo) == 0
    if track == "A":
        m = m.copy()
        m[:, 0] = False
    return m


def _mae(err, m):
    n = m.sum()
    return float(err[m].sum() / n) if n else None


def evaluate(pred, p, uo, R, C, is_pi, track):
    """Return dict: overall MAE, per (R,C) cell, PI vs non-PI, per step index 0..31."""
    m = insp_mask(uo, track)
    err = np.abs(np.asarray(pred, dtype=np.float64) - p)
    R, C, is_pi = np.asarray(R), np.asarray(C), np.asarray(is_pi, dtype=bool)
    out = {"track": track, "n_breaths": int(len(err)), "n_steps": int(m.sum()), "mae": _mae(err, m)}
    out["per_rc"] = {f"R{r}_C{c}": _mae(err, m & ((R == r) & (C == c))[:, None])
                     for r in np.unique(R) for c in np.unique(C)}
    out["pi"] = {"pi": _mae(err, m & is_pi[:, None]), "non_pi": _mae(err, m & ~is_pi[:, None])}
    out["per_step"] = [_mae(err[:, k:k + 1], m[:, k:k + 1]) for k in range(N_STEPS)]
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def save_metrics(run_dir, d):
    """Merge d into run_dir/metrics.json (so train, eval and cost can write separately)."""
    f = Path(run_dir) / "metrics.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur.update(_clean(d))
    f.write_text(json.dumps(cur, indent=1))
    return cur
