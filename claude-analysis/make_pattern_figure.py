"""Figure for pattern_families.json: accuracy by walk shape at fixed hops and points."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("pattern_families.json"))["cells"]
col = {"torn-bigger-13 (Feb 2025)": "#1f5fa8", "g_m10": "#c2571a", "g_plant10": "#3a8a3a"}
lab = {"torn-bigger-13 (Feb 2025)": "torn-bigger-13 (Feb 2025)", "g_m10": "g_m10 (fresh data)", "g_plant10": "g_plant10 (planted chains)"}
fams = ["cycle", "random, fewest reversals", "random, most reversals", "ping-pong"]
ticks = ["cycle", "random,\nfew reversals", "random,\nmany reversals", "ping-pong"]
cells = [(16, 5), (18, 5), (16, 10), (20, 10)]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
for ax, (L, s) in zip(axs, cells):
    for m in col:
        ys = [d[m]["walk"][f"L={L}, s={s}: {f}"]["acc"] for f in fams]
        ax.plot(range(4), ys, "-o", lw=2, ms=5, color=col[m], label=lab[m])
    rv = [d["g_m10"]["walk"][f"L={L}, s={s}: {f}"]["reversal_share"] for f in fams]
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{t}\n({r:.2f})" for t, r in zip(ticks, rv)], fontsize=7)
    ax.set_title(f"{L} hops through {s} points", fontsize=10)
    ax.set_ylim(0.4, 1.02); ax.grid(axis="y", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("accuracy at the slot")
axs[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.text(0.5, -0.04, "walk shape (share of hops that return to the point just left)", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("fig_pattern_families.png", dpi=200, bbox_inches="tight")
