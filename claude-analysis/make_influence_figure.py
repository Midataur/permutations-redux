"""Figure for Test 2: how often replacing ONE letter changes the model's prediction, by reading order (0 = first letter read)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f2328", "#57606a", "#fbfbfa", "#e6e6e3"
M = json.load(open("influence_map.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF, sharey=True)
for ax, L in zip(axes, ("12", "20")):
    ax.set_facecolor(SURF)
    r = np.arange(25)
    hop = np.array([np.nan if v is None else v for v in M[L]["by_position_hop"]])[::-1]
    irr = np.array([np.nan if v is None else v for v in M[L]["by_position_irr"]])[::-1]
    ax.plot(r, 100 * hop, "o-", color=BLUE, ms=4, lw=1.4, label="hop letter replaced (answer changes)")
    ax.plot(r, 100 * irr, "s-", color=ORANGE, ms=4, lw=1.4, label="irrelevant letter replaced (answer unchanged)")
    ax.set_title(f"planted chase with L = {L} hops (s = {int(L)+1})", fontsize=10, color=INK)
    ax.set_xlabel("reading order of the replaced letter (0 = last letter of the word, read first)", fontsize=8.5, color=INK2)
    ax.grid(True, color=GRID, lw=.8); ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("% of cases where the model's prediction changes", fontsize=9, color=INK2)
axes[0].set_ylim(-3, 103); axes[0].legend(frameon=False, fontsize=8, loc="center left")
fig.tight_layout(); fig.savefig("fig_influence_map.png", dpi=130, facecolor=SURF)
print("saved")
