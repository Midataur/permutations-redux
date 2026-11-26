"""Figures for §3.12: attention patterns and single-head ablation, both models.
Reads attention_{general,elem}_results.json and head_ablation_{general,elem}_results.json.
  fig_attention_heads.png     heatmaps over (block, head): next-hop share, previous-hop share, ablation at word positions, ablation on the output side
  fig_attention_profiles.png  hop-offset profiles (word positions) and the query's attention by word position (elementary model)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#0072B2", "#D55E00", "#009E73", "#6b6b6b", "#B8322A", "#7a4fa3"
A = {k: json.load(open(f"attention_{k}_results.json")) for k in ("general", "elem")}
H = {k: json.load(open(f"head_ablation_{k}_results.json")) for k in ("general", "elem")}
LSEL = {"general": "16", "elem": "80"}
NAME = {"general": "General model (`torn-bigger-13`), $L$ = 16", "elem": "Elementary model, $L$ = 80"}


def heat(ax, M, cmap, vmin, vmax, title, fmt, light_above, ylabel=False):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8.5, color="white" if v > light_above else "#222")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(range(M.shape[1]), fontsize=9)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"block {b}" for b in range(1, M.shape[0] + 1)] if ylabel else [""] * M.shape[0], fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)


fig, axs = plt.subplots(2, 4, figsize=(15.5, 8.4))
for r, kind in enumerate(("general", "elem")):
    L = LSEL[kind]; W = np.array(A[kind]["by_L"][L]["word_mass"]); E = A[kind]["by_L"][L]["word_expected"]
    names = A[kind]["categories"]; nx, pv = names.index("next"), names.index("prev")
    heat(axs[r, 0], 100 * W[:, :, nx], "Blues", 0, 60, f"{'AE'[r]}. Attention on the next hop letter (%)\nuniform would give {100*E[nx]:.1f}", lambda v: f"{v:.0f}", 30, ylabel=True)
    heat(axs[r, 1], 100 * W[:, :, pv], "Blues", 0, 60, f"{'BF'[r]}. Attention on the previous hop letter (%)\nuniform would give {100*E[pv]:.1f}", lambda v: f"{v:.0f}", 30)
    ab = H[kind]["by_L"][L]
    heat(axs[r, 2], 100 * (1 - np.array(ab["acc_word"])), "Oranges", 0, 100, f"{'CG'[r]}. Head zeroed at the word positions:\naccuracy lost (points), planted slot", lambda v: f"{v:.0f}", 55)
    heat(axs[r, 3], 100 * (1 - np.array(ab["acc_output"])), "Oranges", 0, 100, f"{'DH'[r]}. Head zeroed at START and outputs:\naccuracy lost (points), planted slot", lambda v: f"{v:.0f}", 55)
fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.035, wspace=0.12, hspace=0.50)
for r, (kind, y) in enumerate((("general", 0.975), ("elem", 0.49))):
    fig.text(0.012, y, NAME[kind].replace("`", "") + "   (columns of each panel: heads 0 to 5)", fontsize=11.5, fontweight="bold", ha="left", va="top")
fig.savefig("fig_attention_heads.png", dpi=170)

# ---- profiles
fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax = axs[0]
r = A["elem"]["by_L"]["80"]; N = A["elem"]["N"]; O = np.array(r["offset_mass"]) * N; offs = np.array(r["offsets"])
sel = [(1, 0, BLUE, "block 1 head 0 (±1)"), (2, 4, GREEN, "block 2 head 4 (±2)"), (3, 5, ORANGE, "block 3 head 5 (+2, +4)"), (3, 2, RED, "block 3 head 2 (−2, −4)"), (4, 1, PURPLE, "block 4 head 1 (+6 to +16)"), (4, 5, GREY, "block 4 head 5 (−8, −6)")]
for l, h, c, lab in sel:
    ax.plot(offs, O[l - 1, h], marker="o", ms=3.5, lw=1.5, color=c, label=lab)
ax.axhline(1, color="#999", lw=.8, ls=":")
ax.set_xlabel("hop offset $d$: the letter at hop $j$ attends to the letter at hop $j+d$"); ax.set_ylabel("attention, in multiples of uniform (1/120)")
ax.set_xticks(range(-8, 17, 4)); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("A. Elementary model, $L$ = 80: where chain letters look", fontsize=10, loc="left")

ax = axs[1]
r = A["general"]["by_L"]["16"]; N = A["general"]["N"]; O = np.array(r["offset_mass"]) * N; offs = np.array(r["offsets"])
sel = [(1, 3, BLUE, "block 1 head 3"), (2, 5, GREEN, "block 2 head 5"), (2, 3, RED, "block 2 head 3"), (3, 3, ORANGE, "block 3 head 3"), (3, 4, PURPLE, "block 3 head 4"), (3, 5, GREY, "block 3 head 5")]
for l, h, c, lab in sel:
    ax.plot(offs, O[l - 1, h], marker="o", ms=3.5, lw=1.5, color=c, label=lab)
ax.axhline(1, color="#999", lw=.8, ls=":")
ax.set_xlabel("hop offset $d$"); ax.set_ylabel("attention, in multiples of uniform (1/25)")
ax.set_xticks(range(-8, 17, 4)); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("B. General model, $L$ = 16", fontsize=10, loc="left")

ax = axs[2]
r = A["elem"]["by_L"]["80"]; N = A["elem"]["N"]
x = np.arange(N)
for (l, h, c, lab) in ((4, 0, BLUE, "block 4 head 0"), (5, 1, ORANGE, "block 5 head 1")):
    ch = np.array(r["query_pos_given_chain"])[l - 1, h] * N; nc = np.array(r["query_pos_given_nonchain"])[l - 1, h] * N
    ax.plot(x, ch, color=c, lw=1.8, label=f"{lab}, chain letters")
    ax.plot(x, nc, color=c, lw=1.4, ls="--", alpha=.8, label=f"{lab}, other letters")
ax.axhline(1, color="#999", lw=.8, ls=":")
ax.set_xlabel("position in the word (0 = left; letters are read from the right)"); ax.set_ylabel("attention from the query, per letter, in multiples of uniform")
ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("C. Elementary model, $L$ = 80: which letters the query reads", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("fig_attention_profiles.png", dpi=170)
print("saved")
