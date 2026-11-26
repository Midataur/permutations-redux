"""Figure for section 3.15: neighbour cosine of the word-position embeddings of the e_base arm (during and at the end of training)
against the original elementary checkpoint.  Usage: python make_ebase_figure.py RUN_DIR [ORIGINAL_MODEL.safetensors]
RUN_DIR = the unpacked hpc-run-<id> folder (needs e_base/pos_emb.npz and e_base/metrics.jsonl).  Writes fig_ebase_blocks.png."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from st_loader import load_safetensors

run = Path(sys.argv[1])
orig_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent.parent / "scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors")
d = np.load(run / "e_base/pos_emb.npz")
steps, pos = d["steps"], d["pos"][:, :120].astype(np.float64)
orig = load_safetensors(orig_path)[0]["position_embedding.weight"].astype(np.float64)[:120]


def nb(P):
    X = P / np.linalg.norm(P, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", X[:-1], X[1:])          # cosine of positions t and t+1


acc = []
for line in open(run / "e_base/metrics.jsonl"):
    r = json.loads(line)
    try:
        acc.append((r["step"], r["eval"]["nat16"]["slot_acc"]))
    except KeyError:
        pass
first = lambda th: next(s for s, v in acc if v >= th)

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6b6b6b"
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
t = np.arange(119)
ax[0].plot(t, nb(orig), color=ORANGE, lw=1.4, label="original elementary checkpoint")
ax[0].plot(t, nb(pos[-1]), color=BLUE, lw=1.4, label=f"e_base, step {int(steps[-1]):,}")
ax[0].axvline(71.5, color=GREY, lw=0.8, ls="--")
ax[0].text(72.5, 0.93, "original block\nboundary (72)", fontsize=8, color=GREY, va="top")
ax[0].set_xlabel("word position t   (cosine between positions t and t+1)")
ax[0].set_ylabel("neighbour cosine")
ax[0].set_ylim(-0.05, 1.0)
ax[0].legend(fontsize=8, loc="upper left", frameon=False)
ax[0].grid(alpha=0.25)
med = np.array([np.median(nb(p)) for p in pos])
ax[1].plot(np.maximum(steps, 1), med, color=BLUE, lw=1.6)
ax[1].axhline(np.median(nb(orig)), color=ORANGE, lw=1.2, ls=":")
ax[1].text(1.3, np.median(nb(orig)) + 0.015, "original checkpoint (median)", color=ORANGE, fontsize=8)
for th, lab in ((0.5, "natural words 50% right (step {})"), (0.9, "natural words 90% right (step {})")):
    s = first(th)
    lab = lab.format(s)
    ax[1].axvline(s, color=GREY, lw=0.8, ls="--" if th == 0.5 else ":", label=lab)
ax[1].legend(fontsize=8, loc="center left", frameon=False)
ax[1].set_xscale("log")
ax[1].set_xlabel("optimiser step")
ax[1].set_ylabel("median neighbour cosine")
ax[1].set_ylim(-0.02, 0.45)
ax[1].grid(alpha=0.25)
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("fig_ebase_blocks.png", dpi=200)
print("ok", "median final", float(med[-1]), "original", float(np.median(nb(orig))))
