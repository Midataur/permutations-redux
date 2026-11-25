"""Schematic of the activation-patching setup of section 3.10 -> fig_patching_setup.png

Worked example with L = 4 hops (the real words have 25 letters and L = 8, 12 or 16):
slot k = 3, hop letters (read right to left) {3,7}, {7,1}, {1,9}, {9,12}; hop 2 is corrupted to {7,20}.
Top: the word.  Bottom: the clean, corrupted and patched runs through the blocks (patch after block l = 3).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, RED, GREEN, GREY, INK = "#1F4E79", "#B8322A", "#1B7F4B", "#9AA3AD", "#222222"
LIGHT_BLUE, LIGHT_RED, LIGHT_GREEN = "#DCE8F4", "#F6DAD7", "#D7EBDD"

fig, ax = plt.subplots(figsize=(11.4, 8.1))
ax.set_xlim(0, 26.4)
ax.set_ylim(-12.4, 12.6)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.0, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, color=INK, lw=1.2, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- A: the word
ax.text(0.0, 12.1, "A   The word: 25 letters, read from right to left (output slot k = 3)", fontsize=12, fontweight="bold", color=BLUE, va="center")
hop = {20: ("{3,7}", "3 \u2192 7"), 14: ("{7,1}", "7 \u2192 1"), 8: ("{1,9}", "1 \u2192 9"), 3: ("{9,12}", "9 \u2192 12")}
bw, by, bh = 0.9, 6.2, 1.5
for t in range(25):
    x = 0.3 + t
    if t == 14:
        box(x, by, bw, bh, LIGHT_RED, RED, 2.0)
    elif t in hop:
        box(x, by, bw, bh, LIGHT_BLUE, BLUE, 1.4)
    else:
        box(x, by, bw, bh, "#F3F4F6", GREY, 0.8)
    ax.text(x + bw / 2, by - 0.45, str(t), ha="center", va="center", fontsize=7.5, color="#555555")
ax.text(0.25, by - 1.1, "position in the word", ha="left", va="center", fontsize=8, color="#555555")
for n, (t, (lab, mv)) in enumerate(hop.items()):
    x = 0.3 + t + bw / 2
    c = RED if t == 14 else BLUE
    ax.text(x, by + bh / 2, lab, ha="center", va="center", fontsize=6.6, color=c, fontweight="bold")
    ax.text(x, by + bh + 0.55, f"hop {n + 1}", ha="center", va="center", fontsize=9, color=c, fontweight="bold")
    ax.text(x, by + bh + 1.2, mv, ha="center", va="center", fontsize=9, color=INK)
arrow(25.2, 10.55, 0.4, 10.55, color=INK, lw=1.0)
ax.text(12.8, 10.75, "reading order: the pointer starts at k = 3 at the right end", ha="center", va="bottom", fontsize=8.5, color=INK)


def bracket(x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y + 0.25, y, y, y + 0.25], color=color, lw=1.4, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - 0.5, text, ha="center", va="center", fontsize=9, color=color)


bracket(0.3, 14.2, by - 1.9, "letters to its left: read later", GREEN)
bracket(15.2, 25.2, by - 1.9, "letters to its right: read earlier", GREEN)
ax.annotate("hop 2 is corrupted: {7,1} \u2192 {7,20}.\nThe pointer now goes 7 \u2192 20 and leaves the chain,\nso the true answer changes from 12 to 20.",
            xy=(14.75, by - 0.75), xytext=(14.75, 1.7), fontsize=9, color=RED, ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2, shrinkA=2, shrinkB=0))

# ---------------------------------------------------------------- B: three runs
ax.text(0.0, 0.2, "B   Three runs of the model, on the same output slot", fontsize=12, fontweight="bold", color=BLUE, va="center")
cols = ["embeddings", "block 1", "block 2", "block 3", "block 4", "block 5", "answer\nat slot k"]
cx = [5.6, 8.8, 12.0, 15.2, 18.4, 21.6, 24.8]
cw_ = 2.5
for c, x in zip(cols, cx):
    ax.text(x, -0.9, c, ha="center", va="center", fontsize=9, color="#444444")
for l_, x in zip(range(6), cx[:6]):
    ax.text(x, -1.75, f"layer $\\ell$ = {l_}", ha="center", va="center", fontsize=8, color="#777777")

rows = [("Clean run", -3.4, BLUE, LIGHT_BLUE, "12", "the original word"),
        ("Patched run", -6.3, GREEN, LIGHT_GREEN, "12 or 20", "the corrupted word, with\nclean values put back"),
        ("Corrupted run", -9.2, RED, LIGHT_RED, "20", "the corrupted word")]
for name, y, ec, fc, ans, sub in rows:
    ax.text(0.0, y + 0.55, name, ha="left", va="center", fontsize=10.5, fontweight="bold", color=ec)
    ax.text(0.0, y + 0.1, sub, ha="left", va="top", fontsize=7.8, color="#555555", linespacing=1.15)
    for i, x in enumerate(cx[:6]):
        e, f = ec, fc
        if name == "Patched run" and i < 3:
            e, f = RED, LIGHT_RED                    # identical to the corrupted run up to the patch
        box(x - cw_ / 2, y - 0.4, cw_, 0.8, f, e, 1.2)
    for i in range(5):
        c = RED if (name == "Patched run" and i < 3) else ec
        arrow(cx[i] + cw_ / 2, y, cx[i + 1] - cw_ / 2, y, color=c, lw=1.0, ms=7)
    arrow(cx[5] + cw_ / 2, y, cx[6] - 1.0, y, color=ec, lw=1.0, ms=7)
    box(cx[6] - 1.0, y - 0.4, 2.0, 0.8, "white", ec, 1.8)
    ax.text(cx[6], y, ans, ha="center", va="center", fontsize=10, fontweight="bold", color=ec)
ax.text(cx[6], -7.2, "12 = restored", ha="center", va="center", fontsize=8, color=GREEN)

xp = cx[3]
arrow(xp, -3.4 - 0.45, xp, -6.3 + 0.5, color=GREEN, lw=2.2, ms=12)
ax.text(xp + 0.35, -4.85, "after block 3, overwrite the chosen word\npositions with the clean run\u2019s values", ha="left", va="center",
        fontsize=8.8, color=GREEN, fontweight="bold", linespacing=1.2)
ax.text(0.0, -11.3, "Only word positions are overwritten, never the query or output positions. Blocks 4 and 5 then compute on the mixture.\n"
        "\u201cRestored\u201d means the patched run answers 12 again. Here the patch is after block 3 (layer $\\ell$ = 3); the experiment tries every layer from 0 to 5 (layer 5 patches nothing).",
        ha="left", va="center", fontsize=8.2, color="#444444", linespacing=1.3)

fig.savefig("fig_patching_setup.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote fig_patching_setup.png")
