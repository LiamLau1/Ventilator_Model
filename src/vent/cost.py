"""Cost metrics, measured identically for every model (same CPU, threads, batch sizes)."""
import copy
import json
import resource
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil
import torch

from .metrics import save_metrics

CPU_THREADS = 4
BATCH = 1024


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def _median_time(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def flops_per_breath(model, x1):
    """FlopCounterMode if it sees any ops (fused LSTM is usually invisible), else the model's analytic count."""
    counted = 0
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x1)
        counted = int(fc.get_total_flops())
    except Exception:
        pass
    analytic = int(model.flops_per_breath(x1.shape[1])) if hasattr(model, "flops_per_breath") else None
    if counted > 0:
        return dict(flops_per_breath=counted, flops_source="FlopCounterMode", flops_analytic=analytic)
    return dict(flops_per_breath=analytic, flops_source="analytic" if analytic else None)


def measure_cost(model, X, run_dir=None):
    """X: float32 (>=BATCH, 80, F) example inputs. Returns the cost dict (also saved to run_dir/metrics.json)."""
    model = copy.deepcopy(model).cpu().eval()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    xb = torch.from_numpy(np.ascontiguousarray(X[:BATCH]))
    x1 = xb[:1]
    cost = dict(n_params=n_params(model), param_mb=n_params(model) * 4 / 1e6,
                cpu_threads=CPU_THREADS, batch=len(xb), **flops_per_breath(model, x1))
    with torch.no_grad():
        # batched latency + peak memory (RSS sampled in a thread; tracemalloc sees only python allocations)
        proc, stop, peak = psutil.Process(), threading.Event(), [0]
        base = proc.memory_info().rss

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.002)

        th = threading.Thread(target=sample)
        tracemalloc.start()
        th.start()
        tb = _median_time(lambda: model(xb), reps=5)
        stop.set()
        th.join()
        cost["tracemalloc_peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        cost["batched_ms_per_batch"] = tb * 1e3
        cost["batched_us_per_breath"] = tb / len(xb) * 1e6
        cost["infer_peak_rss_delta_mb"] = max(peak[0] - base, 0) / 1e6
        cost["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: bytes
        # single-breath latency: true step-by-step streaming if the model has step(), else full sequence
        T = x1.shape[1]
        try:
            def stream():
                state = None
                for k in range(T):
                    _, state = model.step(x1[:, k], state)
            ts = _median_time(stream, reps=20)
            cost.update(streaming_ms_per_breath=ts * 1e3, streaming_us_per_step=ts / T * 1e6,
                        streaming_is_fallback=False)
        except (AttributeError, NotImplementedError):
            ts = _median_time(lambda: model(x1), reps=20)
            cost.update(streaming_ms_per_breath=ts * 1e3, streaming_us_per_step=ts / T * 1e6,
                        streaming_is_fallback=True)
    torch.set_num_threads(prev_threads)
    if run_dir is not None:
        f = Path(run_dir) / "metrics.json"
        tr = json.loads(f.read_text()).get("train", {}) if f.exists() else {}
        cost.update({k: tr[k] for k in ("wall_time_s", "epochs") if k in tr})
        save_metrics(run_dir, {"cost": cost})
    return cost
