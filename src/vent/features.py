"""Per-step input tensors (N, 80, F), float32, for the LSTM variants and tracks."""
import numpy as np

RS, CS = (5, 20, 50), (10, 20, 50)
DT0, TIME0 = 0.035, 3.0
P0_MEAN, P0_STD = 6.0, 3.0


def n_features(track):
    return 10 + (track == "A")


def build_features(d, variant, track, idx=None):
    """Features for variant in {plant, filter, bi}: u/100, u_out, onehot R, onehot C, dt/0.035,
    time/3 (+ constant (p[0]-6)/3 on track A). 'plant' shifts u and u_out one step later
    (zero at k=0), so the input at step k never contains u[k]."""
    sel = slice(None) if idx is None else idx
    t, u, uo = d["t"][sel], d["u"][sel], d["uo"][sel]
    R, C = d["R"][sel], d["C"][sel]
    n, T = u.shape
    dt = np.zeros_like(t)
    dt[:, 1:] = np.diff(t, axis=1)
    uu, oo = u / 100.0, uo.astype(np.float64)
    if variant == "plant":
        uu = np.pad(uu, ((0, 0), (1, 0)))[:, :-1]
        oo = np.pad(oo, ((0, 0), (1, 0)))[:, :-1]
    cols = [uu, oo]
    cols += [np.repeat((R == r)[:, None], T, 1) for r in RS]
    cols += [np.repeat((C == c)[:, None], T, 1) for c in CS]
    cols += [dt / DT0, t / TIME0]
    if track == "A":
        cols.append(np.repeat(((d["p"][sel][:, :1] - P0_MEAN) / P0_STD), T, 1))
    return np.stack(cols, axis=2).astype(np.float32)
