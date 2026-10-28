"""CSV -> compact float64 tensors (cache/*.npz) and the fixed stratified split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
T = 80  # steps per breath

_DTYPES = {"breath_id": np.int32, "R": np.int8, "C": np.int8, "time_step": np.float64,
           "u_in": np.float64, "u_out": np.int8, "pressure": np.float64}


def _tensorise(csv, has_p):
    """Read one csv and return {key: array} with breath-major (N, 80) arrays."""
    cols = [c for c in _DTYPES if has_p or c != "pressure"]
    df = pd.read_csv(csv, usecols=cols, dtype=_DTYPES)
    n = len(df) // T
    assert len(df) == n * T and (df.breath_id.to_numpy().reshape(n, T) == df.breath_id.to_numpy()[::T, None]).all()
    B = lambda c: df[c].to_numpy().reshape(n, T)
    out = dict(t=B("time_step"), u=B("u_in"), uo=B("u_out"), R=B("R")[:, 0].copy(),
               C=B("C")[:, 0].copy(), breath_id=B("breath_id")[:, 0].copy())
    if has_p:
        out["p"] = B("pressure")
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def build_cache():
    """Write cache/train_tensor.npz and cache/test_tensor.npz (float64 t, u, p)."""
    CACHE.mkdir(exist_ok=True)
    for split in ("train", "test"):
        d = _tensorise(DATA / f"{split}.csv", has_p=split == "train")
        np.savez_compressed(CACHE / f"{split}_tensor.npz", **d)
        print(f"{split}: {len(d['R']):,} breaths")


def load(split="train"):
    """Return the cached tensors for 'train' or 'test' as a dict of arrays."""
    f = CACHE / f"{split}_tensor.npz"
    if not f.exists() or np.load(f)["t"].dtype != np.float64:
        build_cache()
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def make_split(seed=42):
    """20% holdout, then 10% of the rest as val, stratified by (R, C). Saves cache/split.npz."""
    d = load("train")
    key = d["R"].astype(int) * 100 + d["C"]
    idx = np.arange(len(key))
    rest, hold = train_test_split(idx, test_size=0.2, stratify=key, random_state=seed)
    train, val = train_test_split(rest, test_size=0.1, stratify=key[rest], random_state=seed)
    s = dict(train=np.sort(train), val=np.sort(val), holdout=np.sort(hold))
    np.savez(CACHE / "split.npz", **s)
    return s


def load_split():
    """Return {train, val, holdout} index arrays (creates them if missing)."""
    if not (CACHE / "split.npz").exists():
        make_split()
    with np.load(CACHE / "split.npz") as z:
        return {k: z[k] for k in z.files}
