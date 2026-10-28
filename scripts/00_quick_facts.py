"""Quick, numerical facts about the dataset used to ground context.md.

Run:  conda run -n datasci_m1 python scripts/00_quick_facts.py
Prints a short report; nothing is written to disk except a cached .npz of the
(breath, 80, feature) tensor for later scripts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

dtypes = {"id": np.int32, "breath_id": np.int32, "R": np.int8, "C": np.int8,
          "time_step": np.float32, "u_in": np.float32, "u_out": np.int8,
          "pressure": np.float32}
df = pd.read_csv(DATA / "train.csv", dtype=dtypes)
nb = df.breath_id.nunique()
assert len(df) == nb * 80
B = lambda c: df[c].to_numpy().reshape(nb, 80)
t, u, uo, p = B("time_step"), B("u_in"), B("u_out"), B("pressure")
R, C = B("R")[:, 0], B("C")[:, 0]
np.savez_compressed(CACHE / "train_tensor.npz", t=t, u=u, uo=uo, p=p, R=R, C=C,
                    breath_id=B("breath_id")[:, 0])

print(f"breaths: {nb:,}  rows: {len(df):,}")
print("R,C counts:\n", pd.crosstab(R, C))

# --- inspiration length & timing -------------------------------------------
insp_len = (uo == 0).sum(1)
dt = np.diff(t, axis=1)
print(f"inspiratory steps: median {np.median(insp_len)}, range {insp_len.min()}-{insp_len.max()}")
print(f"dt: median {np.median(dt):.4f}s, p1 {np.percentile(dt,1):.4f}, p99 {np.percentile(dt,99):.4f}")
# is u_out monotone (single switch)?
print("u_out single switch fraction:", np.mean((np.diff(uo, axis=1) < 0).sum(1) == 0))

# --- pressure discretisation -------------------------------------------------
pu = np.unique(p)
step = np.median(np.diff(pu))
print(f"unique pressures: {len(pu)}, min {pu.min():.4f}, max {pu.max():.4f}, step {step:.6f}")

# --- u_in during inspiration -------------------------------------------------
ui = np.where(uo == 0, u, np.nan)
print(f"insp u_in: frac==0 {np.nanmean(ui==0):.3f}, frac==100 {np.nanmean(ui>=99.999):.3f}")
print("p at t=0: median %.2f, IQR %.2f-%.2f" % tuple(np.percentile(p[:, 0], [50, 25, 75])))

# --- PI-controller test ------------------------------------------------------
# Hypothesis (from 1st-place code): during inspiration
#   e[t] = kt - p[t];  I[t] = I[t-1] + a_t (e[t] - I[t-1]),  a_t = dt/(dt+0.5)
#   u[t] = kp e[t] + ki I[t]
# Eliminating I:  u[t] - (1-a_t) u[t-1] = c0 + c1 p[t] + c2 p[t-1]   (exact, linear)
# Fit per breath on inspiratory, unclipped steps; R^2 ~ 1 => PI-controlled breath.
def pi_r2(i):
    m = (uo[i, 1:] == 0) & (u[i, 1:] > 0) & (u[i, 1:] < 100) & (u[i, :-1] > 0) & (u[i, :-1] < 100)
    m[:1] = False  # skip first step (integrator initialisation)
    if m.sum() < 8:
        return np.nan, np.nan
    a = dt[i] / (dt[i] + 0.5)
    y = (u[i, 1:] - (1 - a) * u[i, :-1])[m]
    X = np.c_[np.ones(m.sum()), p[i, 1:][m], p[i, :-1][m]]
    coef, *_ = np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)
    res = y - X @ coef
    return 1 - res.var() / max(y.var(), 1e-12), np.abs(res).max()

idx = rng.choice(nb, 10_000, replace=False)
r2 = np.array([pi_r2(i) for i in idx])
ok = ~np.isnan(r2[:, 0])
print(f"PI test on {ok.sum()} breaths: frac max|resid|<1e-3: {np.mean(r2[ok,1] < 1e-3):.3f}, "
      f"frac R2>0.999: {np.mean(r2[ok,0] > 0.999):.3f}, median R2 {np.median(r2[ok,0]):.4f}")
print("PI exact-fit frac by R,C:")
exact = pd.Series(r2[:, 1] < 1e-3, index=idx)
print(pd.crosstab(R[idx], C[idx], values=exact.values, aggfunc="mean").round(3))

# --- linear (ARX) plant baselines on inspiration -----------------------------
# Forward plant: p[t] from (p[t-1], u[t-1]) per (R,C)  -> how 'RC-like' is the plant?
# 'Leaky' version adds u[t]: same-step u_in carries controller information about p[t].
def arx_mae(use_ut):
    errs = []
    for r in np.unique(R):
        for c in np.unique(C):
            b = (R == r) & (C == c)
            m = (uo[b, 1:] == 0)
            cols = [p[b, :-1], u[b, :-1], np.ones_like(u[b, :-1])] + ([u[b, 1:]] if use_ut else [])
            X = np.stack([x[m] for x in cols], 1).astype(np.float64)
            y = p[b, 1:][m].astype(np.float64)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            errs.append(np.abs(y - X @ coef))
    return np.concatenate(errs).mean()

print(f"one-step ARX MAE (teacher forced): plant-only {arx_mae(False):.3f} | +u[t] {arx_mae(True):.3f}")
print(f"predict-mean-per-(R,C,t) MAE: ", end="")
key = R.astype(int) * 100 + C
mae = []
for k in np.unique(key):
    pb = p[key == k]
    mae.append(np.abs(pb - pb.mean(0))[uo[key == k] == 0])
print(f"{np.concatenate(mae).mean():.3f}")
