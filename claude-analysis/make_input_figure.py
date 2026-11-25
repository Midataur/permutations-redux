"""Figure for §3.9-3.10: input-position probes (heatmap) and activation patching (lines). Reads input_probe_results.json and patching_results.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = json.load(open("input_probe_results.json"))["by_layer_and_remaining_hops"]
R = json.load(open("patching_results.json"))
bins = list(P["0"].keys())
M = np.array([[P[str(l)][b][0] for b in bins] for l in range(6)])
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
im = ax.imshow(M[::-1], cmap="Blues", vmin=0, vmax=0.85, aspect="auto")
for i in range(6):
    for j in range(len(bins)):
        v = M[::-1][i, j]
        ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 0.45 else "#222")
ax.set_xticks(range(len(bins))); ax.set_xticklabels([(b.split("-")[0] if b.split("-")[0] == b.split("-")[1] else b.replace("-", "–")) for b in bins], fontsize=9)
ax.set_yticks(range(6)); ax.set_yticklabels([f"layer {l}" for l in range(5, -1, -1)], fontsize=9)
ax.set_xlabel("hops still to go after the letter (h)"); ax.set_title("A. Probe at the word positions: accuracy (%), chance 4%", fontsize=10.5, loc="left")
for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)

L = np.arange(5)
ax = axs[1]
for name, key, c, kw in (("last hop corrupted (identical to the line below)", "L12_last", GREEN, dict(lw=4.5, ms=9, alpha=.45)),
                         ("middle hop corrupted", "L12_middle", BLUE, dict(lw=1.8, ms=5, ls="--")),
                         ("first hop corrupted", "L12_first", ORANGE, dict(lw=1.8, ms=5))):
    y = [R[key]["patched"]["all"][l][0] for l in L]
    ax.plot(L, y, marker="o", color=c, label=name, **kw)
ax.set_xticks(L); ax.set_xlabel("layer at which all word positions get the clean values"); ax.set_ylabel("answer restored to the clean one")
ax.set_ylim(-0.03, 1.05); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("B. Patching all word positions (L = 12)", fontsize=10.5, loc="left")

ax = axs[2]
for name, kind, c, kw in (("letters read later (to the left): stays at baseline", "left", ORANGE, dict(lw=4.5, ms=9, alpha=.45)),
                          ("letters read earlier (to the right)", "right", BLUE, dict(lw=1.8, ms=5)),
                          ("the corrupted letter only", "self", GREY, dict(lw=1.8, ms=5, ls="--"))):
    y = [R["L12_middle"]["patched"][kind][l][0] for l in L]
    ax.plot(L, y, marker="o", color=c, label=name, **kw)
ax.set_xticks(L); ax.set_xlabel("layer at which these positions get the clean values")
ax.set_ylim(-0.03, 1.05); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9, loc="center left")
ax.set_title("C. Which positions carry it (L = 12, middle hop corrupted)", fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig("fig_input_positions.png", dpi=170)
print("saved")
