"""Figure for the report: planted-chase accuracy vs hop count (top) and expected number of training
trajectories with >= L hops under two candidate training supports (bottom)."""
import csv
from math import comb, sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)

rows = list(csv.DictReader(open("ladder_results.csv")))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3 (light)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

def pts(pred):
    out = []
    for r in rows:
        if pred(r):
            L, k, n = int(r["L_hops"]), int(r["tf_k_ok"]), int(r["n_tf"])
            lo, hi = wilson(k, n)
            out.append((L, int(r["s_distinct"]), k/n, lo, hi))
    return sorted(out)

ladder = pts(lambda r: r["cell"] == "ladder_spread" and int(r["s_distinct"]) != 12)
ladder_contig = pts(lambda r: r["cell"] == "ladder_contig")
ctrl = pts(lambda r: r["cell"] == "control_manyhops")
s12 = pts(lambda r: int(r["s_distinct"]) == 12 and r["cell"] in ("ladder_spread", "hops_at_s12"))

fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5], hspace=0.12), facecolor=SURF)
for a in (ax, bx):
    a.set_facecolor(SURF)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)

def draw(series, color, marker, filled, label, dx=0.0):
    x = np.array([p[0] for p in series]) + dx
    y = np.array([p[2] for p in series]) * 100
    lo = np.array([p[3] for p in series]) * 100; hi = np.array([p[4] for p in series]) * 100
    ax.vlines(x, lo, hi, color=color, lw=1.2, zorder=2)
    ax.plot(x, y, marker, color=color, mfc=color if filled else SURF, mec=color, mew=1.6, ms=7, lw=0, zorder=3, label=label)

draw(ladder, BLUE, "o", True, "Minimal ladders (L = s − 1 hops)", -0.12)
draw(ladder_contig, BLUE, "o", False, "Same, hops packed at end of word", 0.12)
draw(s12, AQUA, "s", True, "Fixed s = 12 distinct points, more hops", 0.0)
draw(ctrl, ORANGE, "^", True, "Control: L = 20 hops, s = 4…10 distinct points", 0.0)

ax.set_ylim(35, 102); ax.set_ylabel("Accuracy at planted slot (%)", color=INK2, fontsize=10)
ax.set_title("Accuracy falls mainly with chain length (hops), only weakly with distinct points", loc="left", color=INK, fontsize=12, pad=10)
ax.annotate("L = 20 for all four orange points\n(s = 4, 6, 8, 10) and the s = 12 square:\nsimilar accuracy, very different support", xy=(20, 54), xytext=(20.8, 82), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)

Ls = np.arange(2, 25)
def tail(m, L, N=25):
    q = 2*(m-1)/m**2
    return sum(comb(N, j) * q**j * (1-q)**(N-j) for j in range(L, N+1))
WORDS = 8_000_000
for m, ls, lab in [(10, "--", "m = 10 (paper; sensitivity only)"), (16, "-", "m = 16 (checkpoint config; used)")]:
    cnt = np.array([WORDS * m * tail(m, L) for L in Ls])
    bx.semilogy(Ls, np.maximum(cnt, 1e-6), ls, color=INK2, lw=1.4, label=lab)
bx.axhline(1, color=INK2, lw=0.8, ls=":")
bx.text(24.4, 1.4, "1 example", ha="right", fontsize=8, color=INK2)
bx.set_ylim(1e-4, 3e7); bx.set_xlim(2, 24.6); bx.set_xticks(range(2, 25, 2))
bx.set_ylabel("Expected training\ntrajectories with ≥ L hops", color=INK2, fontsize=9)
bx.set_xlabel("Hops L made by the pointer for the planted output slot", color=INK2, fontsize=10)
bx.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
fig.text(0.01, 0.005, "Teacher-forced; 400 words per point; bars are 95% Wilson intervals. Bottom panel assumes 8M training words of 25 uniform letters on m points (assumption, not verified).",
         fontsize=7.5, color=INK2)
fig.savefig("fig_accuracy_vs_hops.png", dpi=150, bbox_inches="tight", facecolor=SURF)
print("saved")
