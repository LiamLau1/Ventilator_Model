"""Train one LSTM baseline (variant x track) and write results/<run>/ (metrics.json incl. cost).

Run:  conda run -n datasci_m1 python scripts/10_train_lstm.py --variant plant --track B
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from vent import data, train
from vent.cost import measure_cost
from vent.features import build_features
from vent.models.lstm import LSTMBaseline

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["plant", "filter", "bi"], default="plant")
ap.add_argument("--track", choices=["A", "B"], default="B")
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch-size", type=int, default=512)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--run-name", default=None)
ap.add_argument("--max-train-breaths", type=int, default=None)
ap.add_argument("--device", default=None, help=f"cpu|mps (default {train.DEFAULT_DEVICE})")
a = ap.parse_args()
run = a.run_name or f"lstm_{a.variant}_{a.track}"

d = data.load("train")
split = data.load_split()
is_pi = np.load(data.CACHE / "pi_labels.npz")["is_pi"]
X = build_features(d, a.variant, a.track)
model = LSTMBaseline(a.variant, a.track, a.hidden, a.layers)
run_dir = ROOT / "results" / run
m = train.fit(model, X, d, split, is_pi, run_dir, a.track, epochs=a.epochs, batch_size=a.batch_size,
              lr=a.lr, patience=a.patience, device=a.device, max_train=a.max_train_breaths)
c = measure_cost(model, X[split["holdout"]], run_dir)
print(f"{run}: holdout MAE {m['mae']:.4f}  params {c['n_params']:,}  "
      f"train {m['train']['wall_time_s']:.0f}s / {m['train']['epochs']} ep  "
      f"batched {c['batched_us_per_breath']:.0f} us/breath  streaming {c['streaming_ms_per_breath']:.2f} ms/breath"
      f"{' (fallback)' if c['streaming_is_fallback'] else ''}")
