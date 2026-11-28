"""Best held-out R^2 for each layer of each model, predicted from the other general models, against controls."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("cross_probe_general.json"))
names = ["torn-bigger-13", "g_m10", "g_plant10"]; col = {"torn-bigger-13": "#1f5fa8", "g_m10": "#c2571a", "g_plant10": "#3a8a3a"}
fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
for r, g in enumerate(("letters", "outputs")):
    for c, tgt in enumerate(names):
        ax = axs[r, c]
        for src in names:
            if src == tgt:
                continue
            M = np.array(d["pairs"][g][f"{src} -> {tgt}"]["matrix"])
            ax.plot(range(6), M.max(0), "-o", ms=4, lw=2, color=col[src], label=f"from {src}")
        ax.plot(range(6), d["controls"][g][tgt], "--", color="#777777", lw=1.5,
                label="from the letters by offset" if g == "letters" else "from token fed in + answer")
        ax.set_title(f"{g}: predicting {tgt}", fontsize=9); ax.set_xlabel("layer predicted", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_ylim(-0.02, 1.02); ax.grid(axis="y", color="#e5e5e5", lw=.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    axs[r, 0].set_ylabel("best held-out $R^2$", fontsize=8)
plt.tight_layout(); plt.savefig("fig_cross_probe_general.png", dpi=200)
