"""Figure for cross_probe.json: R^2 between layers of the general and the elementary model, with controls."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("cross_probe.json"))["groups"]
fig, axs = plt.subplots(2, 3, figsize=(12, 7.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for row, g in enumerate(("letters", "outputs")):
    r = d[g]
    for col, (key, title, xl, yl) in enumerate((("g_to_e", "general → elementary", "elementary layer (predicted)", "general layer (predictor)"),
                                                ("e_to_g", "elementary → general", "general layer (predicted)", "elementary layer (predictor)"))):
        M = np.array(r[key]); ax = axs[row, col]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] > 0.6 else "k")
        ax.set_title(f"{g}: {title}", fontsize=9); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.tick_params(labelsize=7)
    ax = axs[row, 2]
    best_into_e = np.array(r["g_to_e"]).max(0); best_into_g = np.array(r["e_to_g"]).max(0)
    ax.plot(range(6), best_into_e, "-o", color="#1f5fa8", lw=2, ms=4, label="elementary, best general layer")
    ax.plot(range(6), best_into_g, "-o", color="#c2571a", lw=2, ms=4, label="general, best elementary layer")
    ctrl = [c for c in ("letters_by_offset", "fed_and_answer") if f"{c}_to_e" in r][0]
    lab = "letters by relative offset" if ctrl == "letters_by_offset" else "token fed in and answer"
    ax.plot(range(6), r[f"{ctrl}_to_e"], "--", color="#1f5fa8", lw=1.5, label=f"elementary, from {lab}")
    ax.plot(range(6), r[f"{ctrl}_to_g"], "--", color="#c2571a", lw=1.5, label=f"general, from {lab}")
    ax.plot(range(6), r["g_to_e_shuffled"], ":", color="#777777", lw=1.2, label="shuffled pairing")
    ax.set_ylim(-0.05, 1.02); ax.set_xlabel("layer predicted", fontsize=8); ax.set_ylabel("held-out $R^2$", fontsize=8)
    ax.set_title(f"{g}: against controls", fontsize=9); ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.grid(axis="y", color="#dddddd", lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("fig_cross_probe.png", dpi=200)
