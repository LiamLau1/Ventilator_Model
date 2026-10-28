"""Exact PI-controller identification (uses pressure; for evaluation breakdowns only).

On inspiratory steps with u[t], u[t-1] strictly inside (0, 100):
    u[t] - (1-a)u[t-1] = a*kt*(kp+ki) - kp*(p[t] - (1-a)p[t-1]) - ki*a*p[t],   a = dt/(dt+0.5)
which is linear in theta = (kt(kp+ki), kp, ki).
"""
import numpy as np

TOL = 1e-6


def _fit(X, y):
    th = np.linalg.lstsq(X, y, rcond=None)[0]
    return th, np.abs(y - X @ th)


def _label_one(t, u, uo, p, min_steps=8):
    nan = (0.0, np.nan, np.nan, np.nan, False)
    ok = (uo[1:] == 0) & (u[1:] > 0) & (u[1:] < 100) & (u[:-1] > 0) & (u[:-1] < 100)
    if ok.sum() < min_steps:
        return nan
    dt = np.diff(t)[ok]
    a = dt / (dt + 0.5)
    y = u[1:][ok] - (1 - a) * u[:-1][ok]
    X = np.c_[a, -(p[1:][ok] - (1 - a) * p[:-1][ok]), -a * p[1:][ok]]
    th, r = _fit(X, y)
    frac = np.mean(r < TOL)
    if frac < 0.9:  # robust refit on the better half
        keep = r <= np.median(r)
        if keep.sum() >= 3:
            th, r = _fit(X[keep], y[keep])
            r = np.abs(y - X @ th)
            frac = np.mean(r < TOL)
    kp, ki = th[1], th[2]
    kt = th[0] / (kp + ki) if abs(kp + ki) > 1e-9 else np.nan
    return frac, kp, ki, kt, bool(frac >= 0.8)


def label_pi(t, u, uo, p):
    """Label each breath. Returns dict of arrays: frac, kp, ki, kt, is_pi (float64 / bool)."""
    out = np.array([_label_one(t[i], u[i], uo[i], p[i]) for i in range(len(t))], dtype=object)
    return dict(frac=out[:, 0].astype(np.float64), kp=out[:, 1].astype(np.float64),
                ki=out[:, 2].astype(np.float64), kt=out[:, 3].astype(np.float64),
                is_pi=out[:, 4].astype(bool))
