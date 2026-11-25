"""Figure for Test 6a: elementary-transposition model, planted chases. Left: accuracy vs pointer width w (L fixed);
right: accuracy vs hop count L (w = 8), with the fraction of real training slots having at least L hops on the x axis labels."""
import csv
from math import sqrt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

BLUE, ORANGE, AQUA, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f2328", "#57606a", "#fbfbfa", "#e6e6e3"
R = list(csv.DictReader(open("elem_cells_results.csv")))
H = np.load("elem_train_chase_hist.npz")["H2"].astype(float); hop = H.sum(1); tot = H.sum()
frac = lambda L: hop[L:].sum() / tot

def series(name):
    out = []
    for r in R:
        if r["series"].startswith(name):
            k, n = int(r["tf_k_ok"]), int(r["n_tf"]); lo, hi = wilson(k, n)
            out.append((int(r["L_hops"]), int(r["w_width"]), k / n, lo, hi))
    return out

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.22))
for a in (ax, bx):
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=8.5)

for name, col, lab in (("B", BLUE, "L = 24 hops"), ("C", ORANGE, "L = 40 hops")):
    s = series(name)
    w = np.array([x[1] for x in s]); y = 100 * np.array([x[2] for x in s])
    lo = 100 * np.array([x[3] for x in s]); hi = 100 * np.array([x[4] for x in s])
    ax.errorbar(w, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o-", color=col, ms=5, lw=1.5, capsize=3, label=lab)
ax.axvspan(11.5, 16.6, color="#e9e9e6", alpha=.7, lw=0)
ax.text(14.05, 87.4, "beyond the training\nsupport (width ≥ 12)", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_ylim(86, 100.8); ax.set_xlim(3.5, 16.6)
ax.set_xlabel("pointer width w (number of positions spanned by the chase)", fontsize=9, color=INK2)
ax.set_ylabel("teacher-forced slot accuracy (%)", fontsize=9, color=INK2)
ax.set_title("Accuracy falls when the chase spans more positions than training did", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

s = series("A")
L = np.array([x[0] for x in s]); y = 100 * np.array([x[2] for x in s])
lo = 100 * np.array([x[3] for x in s]); hi = 100 * np.array([x[4] for x in s])
bx.errorbar(np.arange(len(L)), y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o-", color=AQUA, ms=5, lw=1.5, capsize=3)
bx.set_xticks(np.arange(len(L)))
bx.set_xticklabels([f"{l}\n({100 * frac(l):.2g}%)" if frac(l) > 0 else f"{l}\n(none)" for l in L], fontsize=8.5)
bx.set_ylim(86, 100.8)
bx.set_xlabel("hops L   (share of real training slots with at least L hops)", fontsize=9, color=INK2)
bx.set_title("No cliff in hop count at fixed width w = 8", fontsize=10, color=INK, loc="left")
bx.tick_params(labelleft=False)
fig.savefig("fig_elem_cells.png", dpi=130, facecolor=SURF, bbox_inches="tight")
print("saved")
