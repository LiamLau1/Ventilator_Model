"""Build the float64 tensor caches (if missing) and the stratified train/val/holdout split.

Run:  conda run -n datasci_m1 python scripts/01_make_split.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from vent import data

if not (data.CACHE / "test_tensor.npz").exists():
    data.build_cache()
d = data.load("train")
print("train t dtype:", d["t"].dtype, {k: v.shape for k, v in d.items()})
s = data.make_split(seed=42)
print({k: len(v) for k, v in s.items()})
key = d["R"].astype(int) * 100 + d["C"]
print("(R,C) share train/val/holdout for (50,10): ",
      [round(float(np.mean(key[s[k]] == 5010)), 4) for k in ("train", "val", "holdout")])
