"""Generic training loop: AdamW + cosine/warmup, masked MAE, early stopping, checkpoint, holdout eval."""
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from .metrics import evaluate, insp_mask, save_metrics

DEFAULT_DEVICE = "cpu"  # set from the CPU-vs-MPS benchmark (see report); override with --device


def masked_mae(pred, p, m):
    return ((pred - p).abs() * m).sum() / m.sum()


@torch.no_grad()
def predict(model, X, device, bs=2048):
    """X: numpy (N, 80, F) float32 -> numpy (N, 80) float32."""
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).to(device)).float().cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def make_optim(model, lr, steps, weight_decay=1e-2, warmup=0.05):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = max(int(warmup * steps), 1)
    f = lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(steps - w, 1)))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_epoch(model, opt, sched, Xt, Pt, Mt, bs, gen, max_batches=None):
    """One shuffled pass over device tensors. Returns mean batch loss."""
    model.train()
    perm = torch.randperm(len(Xt), generator=gen).to(Xt.device)
    tot, nb = torch.zeros((), device=Xt.device), 0
    for i in range(0, len(Xt), bs):
        if max_batches is not None and nb >= max_batches:
            break
        j = perm[i:i + bs]
        loss = masked_mae(model(Xt[j]), Pt[j], Mt[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.detach()
        nb += 1
        if Xt.device.type == "mps" and nb % 5 == 0:
            torch.mps.empty_cache()  # MPS LSTM otherwise grows to OOM (and slows 4x) within ~20 batches
    return float(tot / max(nb, 1))


def fit(model, X, d, split, is_pi, run_dir, track, epochs=30, batch_size=512, lr=2e-3, patience=5,
        device=None, weight_decay=1e-2, seed=0, max_train=None):
    """Train on split['train'], early-stop on split['val'], evaluate best model on split['holdout'].
    X: (N,80,F) float32 inputs for all breaths; d: tensors dict (needs p, uo, R, C)."""
    device = device or DEFAULT_DEVICE
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr = split["train"] if max_train is None else split["train"][:max_train]
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    Xt, Pt = to(X[tr]), to(d["p"][tr].astype(np.float32))
    Mt = to(insp_mask(d["uo"][tr], track).astype(np.float32))
    va = split["val"]
    Mv = insp_mask(d["uo"][va], track)

    model.to(device)
    steps = epochs * math.ceil(len(tr) / batch_size)
    opt, sched = make_optim(model, lr, steps, weight_decay)
    log = run_dir / "log.csv"
    log.write_text("epoch,train_loss,val_mae,lr,epoch_s\n")
    best, best_ep, bad, t_start = float("inf"), 0, 0, time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, opt, sched, Xt, Pt, Mt, batch_size, gen)
        pv = predict(model, X[va], device)
        val = float(np.abs(pv - d["p"][va])[Mv].mean())
        with open(log, "a") as f:
            csv.writer(f).writerow([ep, f"{loss:.5f}", f"{val:.5f}", f"{sched.get_last_lr()[0]:.3e}", f"{time.time() - t0:.1f}"])
        print(f"epoch {ep:3d}  train {loss:.4f}  val {val:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if val < best:
            best, best_ep, bad = val, ep, 0
            torch.save({"state": model.state_dict(), "config": getattr(model, "config", lambda: {})()}, run_dir / "model.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    wall = time.time() - t_start

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["state"])
    ho = split["holdout"]
    pred = predict(model, X[ho], device)
    m = evaluate(pred, d["p"][ho], d["uo"][ho], d["R"][ho], d["C"][ho], is_pi[ho], track)
    m["train"] = dict(wall_time_s=wall, epochs=ep, best_epoch=best_ep, val_mae=best, device=device,
                      n_train_breaths=len(tr), batch_size=batch_size, lr=lr)
    np.savez_compressed(run_dir / "holdout_preds.npz", pred=pred, idx=ho)
    save_metrics(run_dir, m)
    return m
