"""Label PI-controlled breaths with the exact identity; saves cache/pi_labels.npz.

Run:  conda run -n datasci_m1 python scripts/02_label_pi.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from vent import data
from vent.pid import label_pi

d = data.load("train")
t0 = time.time()
lab = label_pi(d["t"], d["u"], d["uo"], d["p"])
np.savez(data.CACHE / "pi_labels.npz", **lab)
print(f"labelled {len(lab['frac']):,} breaths in {time.time() - t0:.0f}s")

pi = lab["is_pi"]
print(f"is_pi: {pi.mean():.3f}   frac>=0.95: {(lab['frac'] >= 0.95).mean():.3f}")
grid_k = np.r_[0, np.arange(1, 11) / 10, np.arange(2, 11)]
grid_t = np.arange(10, 36, 5)
near = lambda x, g: np.abs(x[:, None] - g[None]).min(1) < 1e-3
print(f"on PI breaths: kp on grid {near(lab['kp'][pi], grid_k).mean():.3f}, "
      f"ki on grid {near(lab['ki'][pi], grid_k).mean():.3f}, "
      f"kt on grid {near(lab['kt'][pi][np.isfinite(lab['kt'][pi])], grid_t).mean():.3f}")
print("kp value counts (PI):", pd.Series(lab["kp"][pi].round(1)).value_counts().head(8).to_dict())
print("kt value counts (PI):", pd.Series(lab["kt"][pi].round(0)).value_counts().head(8).to_dict())
print("is_pi share by (R,C):")
print(pd.crosstab(d["R"], d["C"], values=pi, aggfunc="mean").round(3))
