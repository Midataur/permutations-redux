"""
Block structure of the word-position embeddings across the elementary-transposition models in the Spartan model folder.

elem-partitioned-long-boosted-1 (Figure 5b) splits its 120 word positions into two blocks, [0,72) and [72,120), with
the 16 output positions as a third block (report section 3.8).  This script measures, for every model whose position
table has the elementary layout (120 word positions + 16 output positions, i.e. 136 or 137 rows):

  * a boundary detector: for each k, 1 - corr(row k, row k+1) of the cosine matrix over all columns more than 6 from
    both rows, as a robust z-score against all k (median and MAD); a block boundary is a large z; reported: the
    three largest, and z at 72 and 73 (the two-block contrast is kept in the json but is dominated by the ends);
  * the row-change statistic of fig5b_provenance.py: 1 - corr(row k, row k+1) of the cosine matrix over columns
    0-52, for k in 60..119; its largest value and where;
  * the cosine between neighbouring positions: median, value at (71,72), and the smallest value with its position.

Usage: python block_survey.py SNAPSHOT_DIR [EXTRA name=path ...]
Writes block_survey.json, block_survey.txt and fig_block_survey.png.
"""
import os, sys, json
import numpy as np
from np_transformer import load_safetensors

root = sys.argv[1]
extra = dict(a.split("=", 1) for a in sys.argv[2:])
W = 120


def load_pos(path):
    T, _ = load_safetensors(path)
    for k in ("position_embedding.weight", "_orig_mod.position_embedding.weight"):
        if k in T:
            return T[k].astype(np.float64)
    return None


def cosm(P):
    Q = P / np.linalg.norm(P, axis=1, keepdims=True)
    return Q @ Q.T


def measure(P):
    C = cosm(P)
    Cw = C[:W, :W]
    i, j = np.indices(Cw.shape)
    far = np.abs(i - j) > 6
    best = (None, -9)
    contrast = {}
    for b in range(10, 111):
        A = (i < b) & (j < b) & far
        B = (i >= b) & (j >= b) & far
        X = ((i < b) & (j >= b)) | ((i >= b) & (j < b))
        c = 0.5 * (Cw[A].mean() + Cw[B].mean()) - Cw[X].mean()
        contrast[b] = float(c)
        if c > best[1]:
            best = (b, float(c))
    nr = lambda v: (v - v.mean()) / (np.linalg.norm(v - v.mean()) + 1e-9)
    sp = {k + 1: 1 - float(nr(C[k, :53]) @ nr(C[k + 1, :53])) for k in range(60, 119)}
    # boundary detector over the whole table: how much row k and row k+1 differ, over all columns more than 6 from both
    n = C.shape[0]
    dk = {}
    for k in range(5, W - 6):
        J = np.array([j for j in range(n) if abs(j - k) > 6 and abs(j - k - 1) > 6])
        dk[k + 1] = 1 - float(nr(C[k, J]) @ nr(C[k + 1, J]))
    vals = np.array(list(dk.values())); med = np.median(vals); mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-9
    z = {b: (v - med) / mad for b, v in dk.items()}
    top = sorted(z, key=z.get, reverse=True)[:3]
    kmax = max(sp, key=sp.get)
    nb = np.array([C[t, t + 1] for t in range(W - 1)])
    return dict(boundary_top=[(int(b), round(float(z[b]), 1)) for b in top], boundary_z_72=round(float(z[72]), 1),
                boundary_z_73=round(float(z[73]), 1), best_split=best[0], best_contrast=round(best[1], 4), contrast_at_72=round(contrast[72], 4),
                rowchange_max_at=kmax, rowchange_max=round(sp[kmax], 3), rowchange_median=round(float(np.median(list(sp.values()))), 3),
                rowchange_at_72=round(sp[72], 3), rowchange_at_73=round(sp[73], 3),
                nb_median=round(float(np.median(nb)), 3), nb_71=round(float(nb[71]), 3),
                nb_min=round(float(nb.min()), 3), nb_min_at=int(nb.argmin())), C


models = {}
for name in sorted(os.listdir(root)):
    f = os.path.join(root, name, "model.safetensors")
    if os.path.exists(f):
        models[name] = f
models.update(extra)
res, mats = {}, {}
for name, f in models.items():
    P = load_pos(f)
    if P is None or P.shape[0] not in (136, 137):
        continue
    r, C = measure(P)
    r["rows"] = int(P.shape[0])
    res[name] = r; mats[name] = C

lines = [f"{'model':34s} rows  strongest boundaries (robust z)   z@72  z@73   row-change max (median; @72,@73)   neighbour cos median, @71, min@"]
for n, r in res.items():
    tb = "  ".join(f"{b}:{zz:.0f}" for b, zz in r["boundary_top"])
    lines.append(f"{n:34s} {r['rows']:4d}  {tb:30s}  {r['boundary_z_72']:5.1f} {r['boundary_z_73']:5.1f}   "
                 f"{r['rowchange_max']:.2f}@{r['rowchange_max_at']:3d} ({r['rowchange_median']:.2f}; {r['rowchange_at_72']:.2f},{r['rowchange_at_73']:.2f})   "
                 f"{r['nb_median']:.2f}, {r['nb_71']:.2f}, {r['nb_min']:.2f}@{r['nb_min_at']}")
open("block_survey.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
json.dump(res, open("block_survey.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = list(mats)
cols = 6
rows = (len(names) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.75 * rows))
for ax in axs.ravel():
    ax.axis("off")
for ax, n in zip(axs.ravel(), names):
    ax.imshow(mats[n][:136, :136], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axhline(71.5, color="k", lw=0.4, ls=":"); ax.axvline(71.5, color="k", lw=0.4, ls=":")
    ax.set_title(f"{n}\nz@72 {res[n]['boundary_z_72']:.0f}; top {res[n]['boundary_top'][0][0]}", fontsize=7)
plt.tight_layout()
plt.savefig("fig_block_survey.png", dpi=130)
print("wrote fig_block_survey.png")
