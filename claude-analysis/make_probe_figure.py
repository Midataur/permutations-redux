"""Figure for Test 1 (layer-wise probes): heatmaps of probe accuracy (layer x hop index r) and correct-vs-wrong lines."""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#1f2328", "#57606a", "#fbfbfa", "#e6e6e3"
R = list(csv.DictReader(open("probe_results.csv")))
designs = []
for r in R:
    d = (int(r["L"]), int(r["s"]))
    if d not in designs: designs.append(d)
acc = lambda d: np.array([[next(float(x["n_hit"]) / float(x["n_test"]) for x in R
                               if (int(x["L"]), int(x["s"]), int(x["layer"]), int(x["r"])) == (*d, l, rr))
                           for rr in range(d[0] + 1)] for l in range(6)])
fig = plt.figure(figsize=(15, 7.6), facecolor=SURF)
gs = fig.add_gridspec(2, len(designs), height_ratios=[1, 0.9], hspace=0.5, wspace=0.28)
for j, d in enumerate(designs):
    ax = fig.add_subplot(gs[0, j]); A = acc(d)
    im = ax.imshow(A, vmin=0, vmax=1, cmap="Blues", aspect="auto", origin="lower")
    ax.set_title(f"L = {d[0]} hops, s = {d[1]} points", fontsize=10, color=INK)
    ax.set_xlabel("hops r made by the pointer", color=INK2, fontsize=9); ax.set_ylabel("layer (0 = embeddings)" if j == 0 else "", color=INK2, fontsize=9)
    ax.set_yticks(range(6)); ax.set_xticks(range(0, d[0] + 1, 2 if d[0] > 10 else 1)); ax.tick_params(labelsize=8, colors=INK2)
    for s in ax.spines.values(): s.set_visible(False)
cax = fig.add_axes([0.915, 0.57, 0.008, 0.31]); fig.colorbar(im, cax=cax).set_label("probe accuracy (chance 0.04)", color=INK2, fontsize=8)
for j, d in enumerate([x for x in designs if x[0] in (16, 20)][:3]):
    ax = fig.add_subplot(gs[1, j]); ax.set_facecolor(SURF)
    for l, ls in ((2, ":"), (5, "-")):
        for key, col, lab in (("acc_when_model_correct", BLUE, "model correct"), ("acc_when_model_wrong", ORANGE, "model wrong")):
            y = [float(x[key]) if x[key] != "nan" else np.nan for x in R if (int(x["L"]), int(x["s"]), int(x["layer"])) == (*d, l)]
            ax.plot(range(d[0] + 1), y, ls, color=col, lw=1.6, label=f"layer {l}, {lab}")
    ax.set_ylim(0, 1.02); ax.set_title(f"L = {d[0]}, s = {d[1]}: split by model outcome", fontsize=9.5, color=INK)
    ax.set_xlabel("hops r", color=INK2, fontsize=9); ax.set_ylabel("probe accuracy" if j == 0 else "", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=.8); ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if j == 2: ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.savefig("fig_probe_layers.png", dpi=130, bbox_inches="tight", facecolor=SURF)
print("saved")
