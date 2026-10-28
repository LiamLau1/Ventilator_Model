"""Post-hoc analysis of the finished LSTM runs: report1_data.json + figures (no training).

Reads results/lstm_{variant}_{track}/{metrics.json,log.csv,holdout_preds.npz}; runs without
metrics.json are skipped, so re-run when more runs (e.g. bi_A) finish.

Run:  conda run -n datasci_m1 python scripts/11_report_lstm.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vent import data
from vent.metrics import evaluate, insp_mask

OUT = ROOT / "results" / "report1"
FIGS = OUT / "figs"
RUNS = [(v, t) for v in ("plant", "filter", "bi") for t in ("B", "A")]  # missing ones are skipped
QS = [5, 25, 50, 75, 95, 99]
N_STEPS = 32
EDGES = np.arange(0, 70, 5)  # pressure bins 0-5 ... 60-65
COL = {"plant": "#1b9e77", "filter": "#d95f02", "bi": "#7570b3"}
LS = {"A": "--", "B": "-"}


def sig(o, n=4):
    """Round floats to n significant digits, recursively; NaN/inf -> None."""
    if isinstance(o, dict):
        return {k: sig(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sig(v, n) for v in o]
    if isinstance(o, np.ndarray):
        return sig(o.tolist(), n)
    if isinstance(o, np.generic):
        return sig(o.item(), n)
    if isinstance(o, float):
        return float(f"{o:.{n}g}") if np.isfinite(o) else None
    return o


def name(r):
    return f"{r[0]}_{r[1]}"


def mean_masked(x, m, axis=None):
    n = m.sum(axis)
    return np.where(n > 0, (x * m).sum(axis) / np.maximum(n, 1), np.nan) if axis is not None else (
        float((x * m).sum() / n) if n else None)


# ---------------------------------------------------------------- load
d = data.load("train")
ho = data.load_split()["holdout"]
pil = np.load(data.CACHE / "pi_labels.npz")
is_pi, KP, KI, KT = pil["is_pi"], pil["kp"], pil["ki"], pil["kt"]
R, C, uo, P = d["R"][ho], d["C"][ho], d["uo"][ho], d["p"][ho].astype(np.float32)
pi_h = is_pi[ho]

runs, M, PRED, LOG, MASK = [], {}, {}, {}, {}
for r in RUNS:
    rd = ROOT / "results" / f"lstm_{name(r)}"
    if not (rd / "metrics.json").exists():
        print(f"skip {name(r)} (no metrics.json)")
        continue
    runs.append(r)
    M[r] = json.loads((rd / "metrics.json").read_text())
    LOG[r] = pd.read_csv(rd / "log.csv")
    z = np.load(rd / "holdout_preds.npz")
    assert np.array_equal(z["idx"], ho), f"{name(r)}: holdout index order mismatch"
    PRED[r] = z["pred"].astype(np.float32)
    MASK[r] = insp_mask(uo, r[1])
    # verify row order by recomputing the metrics
    ev = evaluate(PRED[r], d["p"][ho], uo, R, C, pi_h, r[1])
    dm = abs(ev["mae"] - M[r]["mae"])
    dpc = max(abs(ev["per_rc"][k] - v) for k, v in M[r]["per_rc"].items())
    dps = max(abs(a - b) for a, b in zip(ev["per_step"], M[r]["per_step"]) if a is not None)
    print(f"verify {name(r):9s} recomputed mae {ev['mae']:.6f} vs json {M[r]['mae']:.6f} "
          f"(|d|={dm:.1e}; per_rc {dpc:.1e}; per_step {dps:.1e})")
    assert max(dm, dpc, dps) < 1e-4, f"{name(r)} MAE mismatch"

ERR = {r: np.abs(PRED[r] - P) for r in runs}
BERR = {r: mean_masked(ERR[r], MASK[r], axis=1) for r in runs}  # per-breath inspiratory MAE
FA, FB, BB = ("filter", "A"), ("filter", "B"), ("bi", "B")

# ---------------------------------------------------------------- 1-3. runs, curves, quantiles
out = {"runs": {}, "curves": {}, "breath_error_quantiles": {}}
for r in runs:
    m, lg, c = M[r], LOG[r], M[r]["cost"]
    out["runs"][name(r)] = dict(
        variant=r[0], track=r[1], mae=m["mae"], pi_mae=m["pi"]["pi"], non_pi_mae=m["pi"]["non_pi"],
        per_rc=m["per_rc"], per_step=m["per_step"], n_params=c["n_params"],
        flops_per_breath=c["flops_per_breath"], batched_us_per_breath=c["batched_us_per_breath"],
        streaming_ms_per_breath=None if c.get("streaming_is_fallback") else c.get("streaming_ms_per_breath"),
        train_minutes=float(lg.epoch_s.sum() / 60), epochs=int(len(lg)), best_epoch=m["train"]["best_epoch"],
        final_val_mae=float(lg.val_mae.iloc[-1]), best_val_mae=m["train"]["val_mae"])
    out["curves"][name(r)] = dict(epoch=lg.epoch.tolist(), train_loss=lg.train_loss.tolist(),
                                  val_mae=lg.val_mae.tolist())
    out["breath_error_quantiles"][name(r)] = dict(zip(map(str, QS), np.percentile(BERR[r], QS).tolist()))

# ---------------------------------------------------------------- 4. examples
rc = lambda a, b: np.flatnonzero((R == a) & (C == b))
used = set()


def closest(cands, target, key=None):
    cands = np.array([c for c in cands if c not in used])
    c = cands[np.argmin(np.abs(BERR[FA][cands] - target))]
    used.add(int(c))
    return int(c)


allb = np.arange(len(ho))
qa = lambda q: float(np.percentile(BERR[FA], q))
med = qa(50)
picks = [(closest(allb, qa(50)), "median-error breath (filter_A)"),
         (closest(allb, qa(90)), "90th-percentile error (filter_A)"),
         (closest(allb, qa(99)), "99th-percentile error (filter_A)"),
         (closest(rc(5, 50), med), "typical R5_C50 (easiest cell)"),
         (closest(rc(50, 10), med), "typical R50_C10 (hardest cell)"),
         (closest(np.flatnonzero(pi_h), med), "typical PI breath"),
         (closest(np.flatnonzero(~pi_h), med), "typical non-PI breath")]
if BB in runs and FB in runs:
    lo, hi = np.percentile(BERR[FB], [40, 60])
    band = np.flatnonzero((BERR[FB] >= lo) & (BERR[FB] <= hi))
    gain = BERR[FB][band] - BERR[BB][band]
    for j in np.argsort(-gain):
        if int(band[j]) not in used:
            used.add(int(band[j]))
            picks.append((int(band[j]), f"bi_B beats filter_B by {gain[j]:.2f} (filter_B near median)"))
            break
EX = []
for b, why in picks:
    g = int(ho[b])
    e = dict(breath=g, holdout_row=b, R=int(R[b]), C=int(C[b]), is_pi=bool(pi_h[b]), why=why,
             err_filter_A=BERR[FA][b],
             t=d["t"][g], u_in=d["u"][g], u_out=d["uo"][g].astype(int), p=d["p"][g])
    if pi_h[b]:
        e.update(kp=KP[g], ki=KI[g], kt=KT[g])
    e["pred"] = {name(r): PRED[r][b].astype(float) for r in runs}
    EX.append(e)
out["examples"] = [sig(e, 6) for e in EX]  # keep 3-4 dp on arrays via rounding below
for e in out["examples"]:
    e["t"] = np.round(e["t"], 4).tolist()
    for k in ("u_in", "p"):
        e[k] = np.round(np.array(e[k]), 3).tolist()
    e["pred"] = {k: np.round(np.array(v), 3).tolist() for k, v in e["pred"].items()}

# ---------------------------------------------------------------- 5. diagnostics
dg = {}
# a. per-step signed bias
dg["bias_per_step"] = {}
for r in runs:
    s, m = (PRED[r] - P)[:, :N_STEPS], MASK[r][:, :N_STEPS]
    dg["bias_per_step"][name(r)] = [None if m[:, k].sum() == 0 else float(s[:, k][m[:, k]].mean())
                                    for k in range(N_STEPS)]
# b. MAE by true-pressure bin
cent = EDGES[:-1] + 2.5
dg["mae_by_pressure_bin"] = {"bin_lo": EDGES[:-1].tolist(), "bin_hi": EDGES[1:].tolist()}
bidx = np.clip(np.digitize(P, EDGES) - 1, 0, len(EDGES) - 2)  # p<0 -> first bin
for r in runs:
    mae, cnt = [], []
    for b in range(len(EDGES) - 1):
        s = MASK[r] & (bidx == b)
        cnt.append(int(s.sum()))
        mae.append(float(ERR[r][s].mean()) if s.any() else None)
    dg["mae_by_pressure_bin"][name(r)] = dict(mae=mae, count=cnt)
# c. MAE vs PI target kt
ktv = KT[ho]
okpi = pi_h & np.isfinite(ktv)
kti = np.round(np.where(np.isfinite(ktv), ktv, -1)).astype(int)
kts = sorted(set(kti[okpi].tolist()))
dg["mae_by_kt"] = {"kt": kts, "n_breaths": [int((okpi & (kti == k)).sum()) for k in kts]}
for r in (FA, BB, ("plant", "B")):
    if r in runs:
        dg["mae_by_kt"][name(r)] = [float(ERR[r][MASK[r] & (okpi & (kti == k))[:, None]].mean()) for k in kts]
# d. early (first 5 steps) vs remainder
k5 = np.arange(80) < 5
dg["early_vs_rest"] = {"note": "step index <5 vs >=5 among scored steps (track A excludes step 0)"}
for grp, gm in (("pi", pi_h), ("non_pi", ~pi_h)):
    dg["early_vs_rest"][grp] = {}
    for r in runs:
        e0 = MASK[r] & gm[:, None] & k5[None]
        e1 = MASK[r] & gm[:, None] & ~k5[None]
        dg["early_vs_rest"][grp][name(r)] = dict(first5=float(ERR[r][e0].mean()), rest=float(ERR[r][e1].mean()))
# e. track A gain
dg["trackA_gain_per_step"] = {}
for v in ("plant", "filter", "bi"):
    if (v, "A") in runs and (v, "B") in runs:
        a, b = M[(v, "A")]["per_step"], M[(v, "B")]["per_step"]
        dg["trackA_gain_per_step"][v] = dict(
            diff_B_minus_A=[None if x is None or y is None else y - x for x, y in zip(a, b)],
            overall_B_minus_A=M[(v, "B")]["mae"] - M[(v, "A")]["mae"])
out["diagnostics"] = dg

OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(exist_ok=True)
js = json.dumps(sig(out), separators=(",", ":"))
(OUT / "report1_data.json").write_text(js)
print(f"wrote report1_data.json  {len(js) / 1e3:.0f} kB")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False})
sty = lambda r: dict(color=COL[r[0]], ls=LS[r[1]], lw=1.6, label=name(r))
save = lambda fig, f: (fig.savefig(FIGS / f, bbox_inches="tight"), plt.close(fig))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for r in runs:
    c = LOG[r]
    ax[0].plot(c.epoch, c.val_mae, **sty(r))
    ax[1].plot(c.epoch, c.train_loss, **sty(r))
ax[0].set(xlabel="epoch", ylabel="validation MAE (cmH$_2$O)", yscale="log", title="Validation MAE")
ax[1].set(xlabel="epoch", ylabel="training loss", yscale="log", title="Training loss")
ax[0].legend(ncol=2)
save(fig, "learning_curves.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
for r in runs:
    y = np.array([np.nan if v is None else v for v in M[r]["per_step"]])
    ax.plot(range(N_STEPS), y, **sty(r))
ax.set(xlabel="inspiratory step", ylabel="MAE (cmH$_2$O)", title="Holdout MAE by inspiratory step")
ax.legend(ncol=2)
save(fig, "per_step_mae.png")

rs, cs = [5, 20, 50], [10, 20, 50]
allv = [v for r in runs for v in M[r]["per_rc"].values()]
vmin, vmax = min(allv), max(allv)
nc = 3
nr = int(np.ceil(len(runs) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.0 * nr), squeeze=False)
for ax in axs.ravel():
    ax.axis("off")
for ax, r in zip(axs.ravel(), runs):
    ax.axis("on")
    ax.grid(False)
    g = np.array([[M[r]["per_rc"][f"R{a}_C{b}"] for b in cs] for a in rs])
    im = ax.imshow(g, vmin=vmin, vmax=vmax, cmap="viridis_r")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if g[i, j] > (vmin + vmax) / 2 else "k")
    ax.set(xticks=range(3), xticklabels=cs, yticks=range(3), yticklabels=rs, xlabel="C", ylabel="R",
           title=f"{name(r)}  (MAE {M[r]['mae']:.3f})")
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label="MAE (cmH$_2$O)")
save(fig, "per_rc_heatmaps.png")

nex = len(EX)
fig, axs = plt.subplots(int(np.ceil(nex / 2)), 2, figsize=(11, 2.9 * np.ceil(nex / 2)), squeeze=False)
shown = [FA, FB, BB, ("plant", "B")]
for ax, e in zip(axs.ravel(), EX):
    t, uo_e = e["t"] - e["t"][0], e["u_out"]
    nz = np.flatnonzero(uo_e == 1)
    if len(nz):
        ax.axvspan(t[nz[0]], t[-1], color="0.9", zorder=0)
    ax.plot(t, e["p"], "k", lw=2, label="true p")
    for r in shown:
        if name(r) in e["pred"]:
            ax.plot(t, e["pred"][name(r)], color=COL[r[0]], ls=LS[r[1]], lw=1.2, label=name(r))
    ins = uo_e == 0  # y-range from truth + scored steps; unscored expiratory preds may run off-axis
    ys = np.concatenate([e["p"]] + [v[ins] for v in e["pred"].values()])
    ax.set_ylim(ys.min() - 1, ys.max() + 2)
    ax2 = ax.twinx()
    ax2.plot(t, e["u_in"], color="tab:blue", alpha=0.35, lw=1, label="u_in")
    ax2.set_ylim(0, 160)
    ax2.set_yticks([0, 50, 100])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylabel("u_in (%)", color="tab:blue", fontsize=8)
    pi = f"PI kt={e['kt']:.0f}" if e["is_pi"] and np.isfinite(e["kt"]) else ("PI" if e["is_pi"] else "non-PI")
    ax.set_title(f"{e['why']}\nR{e['R']} C{e['C']}, {pi}, breath {e['breath']}", fontsize=8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("p (cmH$_2$O)")
h, l = axs.ravel()[0].get_legend_handles_labels()
for ax in axs.ravel()[nex:]:
    ax.axis("off")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.legend(h, l, ncol=len(l), loc="upper center", fontsize=8)
save(fig, "examples.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.axhline(0, color="k", lw=0.8)
for r in runs:
    y = np.array([np.nan if v is None else v for v in dg["bias_per_step"][name(r)]])
    ax.plot(range(N_STEPS), y, **sty(r))
ax.set(xlabel="inspiratory step", ylabel="mean (pred - p)  (cmH$_2$O)", title="Signed bias by inspiratory step")
ax.legend(ncol=2)
save(fig, "bias_per_step.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1.2]})
for r in runs:
    y = np.array([np.nan if v is None else v for v in dg["mae_by_pressure_bin"][name(r)]["mae"]])
    ax[0].plot(cent, y, marker="o", ms=3, **sty(r))
ax[0].set(xlabel="true pressure bin centre (cmH$_2$O)", ylabel="MAE (cmH$_2$O)", title="MAE vs true pressure")
ax[0].legend(ncol=2)
ax[1].bar(cent, dg["mae_by_pressure_bin"][name(FA)]["count"], width=4.5, color="0.6")
ax[1].set(xlabel="true pressure (cmH$_2$O)", ylabel="scored steps (filter_A)", yscale="log", title="Support")
fig.tight_layout()
save(fig, "mae_vs_pressure.png")

# ---------------------------------------------------------------- printed summary
print("\n== JSON keys / sizes (kB) ==")
for k, v in out.items():
    print(f"  {k}: {len(json.dumps(sig(v), separators=(',', ':'))) / 1e3:.1f}")
print("  diagnostics subkeys:", list(dg))
f2 = lambda v: "  -  " if v is None else f"{v:6.3f}"
print("\n== runs ==")
for k, v in out["runs"].items():
    print(f" {k:9s} mae {v['mae']:.4f} pi {v['pi_mae']:.4f} nonpi {v['non_pi_mae']:.4f} params {v['n_params']:,} "
          f"MFLOP {v['flops_per_breath'] / 1e6:.1f} batched {v['batched_us_per_breath']:.0f}us stream "
          f"{v['streaming_ms_per_breath']} train {v['train_minutes']:.1f}min ep {v['epochs']} best {v['best_epoch']} "
          f"finalval {v['final_val_mae']:.4f}")
print("\n== breath error quantiles (5,25,50,75,95,99) ==")
for k, v in out["breath_error_quantiles"].items():
    print(f" {k:9s}", " ".join(f"{x:6.3f}" for x in v.values()))
print("\n== examples ==")
for e in EX:
    print(f" breath {e['breath']:6d} R{e['R']:<2d} C{e['C']:<2d} pi={e['is_pi']!s:5s} errFA {e['err_filter_A']:.3f} | {e['why']}")
print("\n== a. bias per step (steps 0..31) ==")
for r in runs:
    print(f" {name(r):9s}", " ".join(f2(v) for v in dg["bias_per_step"][name(r)]))
print("\n== b. MAE by pressure bin (count of first run in brackets) ==")
print(" bin       ", " ".join(f"{lo:>5d}" for lo in EDGES[:-1]))
for r in runs:
    x = dg["mae_by_pressure_bin"][name(r)]
    print(f" {name(r):9s}", " ".join(f2(v)[:5].rjust(5) for v in x["mae"]), "\n  n        ",
          " ".join(f"{c:>5d}" for c in x["count"]))
print("\n== c. MAE by PI target kt (PI breaths) ==")
print(" kt      ", kts, "\n n_breaths", dg["mae_by_kt"]["n_breaths"])
for r in (FA, BB, ("plant", "B")):
    if r in runs:
        print(f" {name(r):9s}", " ".join(f2(v) for v in dg["mae_by_kt"][name(r)]))
print("\n== d. first 5 steps vs rest ==")
for grp in ("pi", "non_pi"):
    for r in runs:
        x = dg["early_vs_rest"][grp][name(r)]
        print(f" {grp:6s} {name(r):9s} first5 {x['first5']:.3f}  rest {x['rest']:.3f}")
print("\n== e. track gain (B - A) per step ==")
for v, x in dg["trackA_gain_per_step"].items():
    print(f" {v:7s} overall {x['overall_B_minus_A']:+.4f}\n   ", " ".join(f2(q) for q in x["diff_B_minus_A"]))
print("\nfigs:", sorted(p.name for p in FIGS.glob("*.png")))
