"""Pareto plot: match accuracy against program size for the stage-1 graphs of torn-bigger-13 (and the elementary model)."""
import json, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lines import count_lines
pts = []
for f in sorted(glob.glob("runs/general_lin_l*.json")):
    r = json.load(open(f)); k = [(tuple(d), s) for d, s in r["kept"]]
    pts.append((r["lamb"], r["edges_kept"], r["match_word"], r["match_slot"], count_lines(k), count_lines(k, split_mlps=True)))
pts.sort(key=lambda p: p[4])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, idx, lab, full in ((axs[0], 4, "D-RASP lines, each MLP one line", 165818738), (axs[1], 5, "D-RASP lines, MLPs split by input", 35602724216)):
    xs = [p[idx] for p in pts]
    ax.plot(xs, [p[2] for p in pts], "-o", color="#1f5fa8", lw=2, ms=5, label="words: all 25 outputs match")
    ax.plot(xs, [p[3] for p in pts], "--o", color="#6f9fd8", lw=1.5, ms=4, label="single output slots match")
    for p in pts:
        ax.annotate(f"λ={p[0]:g}\n{p[1]} edges", (p[idx], p[2]), textcoords="offset points", xytext=(4, -22), fontsize=6.5, color="#444444")
    ax.axhline(0.9, color="#c2571a", lw=1, ls=":"); ax.text(xs[0], 0.905, "paper's bar (0.9)", fontsize=7, color="#c2571a")
    ax.set_xscale("log"); ax.set_ylim(0.35, 1.02); ax.set_xlabel(lab + f"  (full model: {full:.2g})", fontsize=8)
    ax.set_ylabel("match accuracy", fontsize=8); ax.tick_params(labelsize=7); ax.grid(color="#e5e5e5", lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].legend(fontsize=7, frameon=False, loc="lower right")
plt.suptitle("torn-bigger-13 after stage-1 pruning (linear LayerNorm, one constant per position)", fontsize=9)
plt.tight_layout(); plt.savefig("../fig_decompile.png", dpi=200)
print([(p[0], p[1], round(p[2], 3), round(p[3], 4), p[4], p[5]) for p in pts])
