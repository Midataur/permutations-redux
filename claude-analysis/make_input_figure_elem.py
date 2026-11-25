"""Figure for §3.11: the elementary model's word-position probes (heatmap, natural words) and activation patching (lines).
Reads input_probe_elem_results.json and patching_elem_results.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PR = json.load(open("input_probe_elem_results.json"))
P = PR["by_source"]["natural"]
R = json.load(open("patching_elem_results.json"))
bins = [b for b in P["0"].keys() if P["0"][b][1] >= 50]
M = np.array([[P[str(l)][b][0] for b in bins] for l in range(6)])
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#6b6b6b", "#B8322A"

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
im = ax.imshow(M[::-1], cmap="Blues", vmin=0, vmax=0.95, aspect="auto")
for i in range(6):
    for j in range(len(bins)):
        v = M[::-1][i, j]
        ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 0.45 else "#222")
ax.set_xticks(range(len(bins))); ax.set_xticklabels([(b.split("-")[0] if b.split("-")[0] == b.split("-")[1] else b.replace("-", "–")) for b in bins], fontsize=9)
ax.set_yticks(range(6)); ax.set_yticklabels([f"layer {l}" for l in range(5, -1, -1)], fontsize=9)
ax.set_xlabel("hops still to go after the letter (h)"); ax.set_title("A. Probe at the word positions, natural words: accuracy (%), chance 6%", fontsize=10.5, loc="left")
for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)

L = np.arange(5)
ax = axs[1]
cols = {12: "#9ecae1", 24: "#4292c6", 40: "#08519c", 80: "#08306b"}
for Lh in (12, 24, 40, 80):
    ax.plot(L, [R[f"L{Lh}_first"]["patched"]["all"][l][0] for l in L], marker="o", color=ORANGE, alpha={12: .35, 24: .55, 40: .8, 80: 1.0}[Lh],
            lw=1.8, ms=5, label=f"first hop corrupted, $L$ = {Lh}")
mid = [min(R[f"L{Lh}_{w}"]["patched"]["all"][l][0] for Lh in (12, 24, 40, 80) for w in ("middle", "last")) for l in L]
ax.plot(L, mid, marker="o", color=GREEN, lw=4.5, ms=9, alpha=.45, label="middle or last hop, any $L$ (lowest of the 8 curves)")
ax.set_xticks(L); ax.set_xlabel("layer at which all word positions get the clean values"); ax.set_ylabel("answer restored to the clean one")
ax.set_ylim(-0.03, 1.05); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_title("B. Patching all word positions", fontsize=10.5, loc="left")

ax = axs[2]
for name, kind, c, kw in (("letters to the left (read later)", "left", ORANGE, dict(lw=1.8, ms=5)),
                          ("letters to the right (read earlier)", "right", BLUE, dict(lw=1.8, ms=5)),
                          ("the corrupted letter only", "self", GREY, dict(lw=1.8, ms=5, ls="--")),
                          ("neither chain (control)", "nonchain", RED, dict(lw=1.8, ms=5, ls=":"))):
    ax.plot(L, [R["L40_last"]["patched"][kind][l][0] for l in L], marker="o", color=c, label=name, **kw)
ax.set_xticks(L); ax.set_xlabel("layer at which these positions get the clean values")
ax.set_ylim(-0.03, 1.42); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
ax.set_title("C. Which positions carry it ($L$ = 40, last hop corrupted)", fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig("fig_input_positions_elem.png", dpi=170)
print("saved")
