"""Figure for report section 3.17: training against fresh slots by hop count (Test 16 slots, four models), and the hops x
points grid of torn-bigger-13 and the fresh-data control.  Reads memorisation_results*.json and hop_points_grid.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = {"torn-bigger-13 (Feb 2025)": "", "torn-bigger-13 (resumed)": "_torn13_resumed",
     "g_m10 (fresh data)": "_g_m10", "g_plant10 (planted chains)": "_g_plant10"}
col = {"torn-bigger-13 (Feb 2025)": "#1f5fa8", "torn-bigger-13 (resumed)": "#6f9fd8",
       "g_m10 (fresh data)": "#c2571a", "g_plant10 (planted chains)": "#3a8a3a"}
hs = [12, 13, 14, 15, 16, 17]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
for name, suf in M.items():
    r = json.load(open(f"memorisation_results{suf}.json"))
    ax.plot(hs, [r["fresh"][str(h)]["acc"] for h in hs], "-o", ms=4, lw=2, color=col[name], label=f"{name}, fresh slots")
r = json.load(open("memorisation_results.json"))
ax.plot([h for h in hs if str(h) in r["train"]], [r["train"][str(h)]["acc"] for h in hs if str(h) in r["train"]], "--s", ms=4, lw=1.5,
        color="#1f5fa8", label="torn-bigger-13 (Feb 2025), its own training slots")
ax.set_xlabel("hops h of the slot's chain"); ax.set_ylabel("accuracy at the slot (teacher-forced)")
ax.set_ylim(0.65, 1.01); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left"); ax.set_title("Natural slots by hop count", fontsize=9)
G = json.load(open("hop_points_grid.json"))
Ls, Ss = [10, 12, 13, 14, 15, 16, 17, 18, 20, 22], [3, 5, 7, 10]
for ax, name in zip(axs[1:], ["torn-bigger-13 (Feb 2025)", "g_m10 (fresh-data control)"]):
    A = np.array([[G[name]["planted"].get(f"{L},{s}", np.nan) for s in Ss] for L in Ls])
    im = ax.imshow(A, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto", origin="lower")
    for i in range(len(Ls)):
        for j in range(len(Ss)):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if A[i, j] < 0.75 else "k")
    ax.set_xticks(range(len(Ss))); ax.set_xticklabels(Ss); ax.set_yticks(range(len(Ls))); ax.set_yticklabels(Ls)
    ax.set_xlabel("distinct points s"); ax.set_ylabel("hops L"); ax.set_title(f"Planted chains: {name}", fontsize=9)
fig.colorbar(im, ax=axs[1:], fraction=0.03, label="accuracy")
plt.savefig("fig_memorisation.png", dpi=150, bbox_inches="tight")
print("wrote fig_memorisation.png")
