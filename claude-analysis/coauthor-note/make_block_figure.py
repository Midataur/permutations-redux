"""Compact block-structure figure for the co-author note: four runs with Figure 5b's config."""
import json, sys, numpy as np
sys.path.insert(0, "/home/claude/permutations-redux/claude-analysis")
from np_transformer import load_safetensors
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/home/claude/snapshots-extracted/model-snapshots/"
names = [("elem-partitioned-long-boosted-1", "boosted-1 (Fig. 5b), seed 42", 72), ("elem-partitioned-long-boosted-2", "boosted-2, seed 43", 81),
         ("rerun-3", "rerun-3, seed 43", 78), ("elem-partitioned-long-boosted-4", "boosted-4, seed 97", None)]
fig, axs = plt.subplots(2, 4, figsize=(11, 4.6), gridspec_kw=dict(height_ratios=[3, 1.1]))
for i, (n, title, b) in enumerate(names):
    T, _ = load_safetensors(R + n + "/model.safetensors")
    P = [T[k] for k in T if k.endswith("position_embedding.weight")][0].astype(float)
    Q = P / np.linalg.norm(P, axis=1, keepdims=True); C = Q @ Q.T
    ax = axs[0, i]; ax.imshow(C, cmap="RdBu_r", vmin=-.6, vmax=.6)
    if b:
        ax.axhline(b - .5, color="#1b7a1b", lw=.8); ax.axvline(b - .5, color="#1b7a1b", lw=.8)
    ax.axhline(119.5, color="k", lw=.5); ax.axvline(119.5, color="k", lw=.5)
    ax.set_title(title, fontsize=9); ax.set_xticks([0, 60, 120]); ax.set_yticks([0, 60, 120]); ax.tick_params(labelsize=7)
    nr = lambda v: (v - v.mean()) / (np.linalg.norm(v - v.mean()) + 1e-9); ks = range(5, 114); d = []
    for k in ks:
        J = np.array([j for j in range(C.shape[0]) if abs(j - k) > 6 and abs(j - k - 1) > 6]); d.append(1 - nr(C[k, J]) @ nr(C[k + 1, J]))
    d = np.array(d); z = (d - np.median(d)) / (1.4826 * np.median(np.abs(d - np.median(d))))
    a2 = axs[1, i]; a2.plot([k + 1 for k in ks], z, lw=1, color="#1f5fa8"); a2.set_ylim(-3, 9); a2.tick_params(labelsize=7)
    a2.axhline(4, color="#999999", lw=.6, ls=":"); a2.set_xlabel("word position $b$", fontsize=8)
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
axs[1, 0].set_ylabel("boundary score $z$", fontsize=8)
plt.tight_layout(); plt.savefig("coauthor-note/fig_blocks.png", dpi=200)
