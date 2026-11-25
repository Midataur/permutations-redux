"""
Self-similarity of the learned word-position embeddings (positions 0-119) of the elementary checkpoint
elem-partitioned-long-boosted-1, raw and mean-removed cosine, plus a few numbers about local smoothness and
whether position 45 is special. Writes elem_pos_gram.png.
Usage: python elem_pos_gram.py [path/to/model.safetensors]
"""
import sys
from pathlib import Path
import numpy as np
from st_loader import load_safetensors

default = Path(__file__).resolve().parent.parent / "scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
W, _ = load_safetensors(sys.argv[1] if len(sys.argv) > 1 else str(default))
P = W["position_embedding.weight"].astype(np.float64)[:120]
cos = lambda X: (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T
G, Gc = cos(P), cos(P - P.mean(0))
n = np.linalg.norm(P, axis=1)
print(f"row norms: min {n.min():.2f} mean {n.mean():.2f} max {n.max():.2f}")
print("mean cosine at offset d:", {d: round(float(np.mean(np.diag(G, d))), 3) for d in (1, 2, 5, 10, 20, 45)})
print("mean-removed:", {d: round(float(np.mean(np.diag(Gc, d))), 3) for d in (1, 2, 5, 10, 20, 45)})


def off(a0, a1, b0, b1):
    return float(np.mean([Gc[i, j] for i in range(a0, a1) for j in range(b0, b1) if abs(i - j) >= 8]))


segs = [(0, 45), (45, 72), (72, 120)]
print("off-band mean similarity (|i-j|>=8) between segments [0,45) [45,72) [72,120):")
for s in segs:
    print("  ", [round(off(*s, *t), 3) for t in segs])
sc = sorted(((np.mean([off(0, c, 0, c), off(c, 120, c, 120)]) - off(0, c, c, 120), c) for c in range(15, 106)), reverse=True)
print("best block boundaries (within minus cross):", [(c, round(v, 3)) for v, c in sc[:5]], " at 45:", round(float([v for v, c in sc if c == 45][0]), 3))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
for a, M, t in ((ax[0], G, "cosine"), (ax[1], Gc, "mean-removed cosine")):
    im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1); a.set_title("position embeddings, word positions 0-119: " + t, fontsize=9)
    a.axvline(44.5, color="k", lw=.5); a.axhline(44.5, color="k", lw=.5)
fig.colorbar(im, ax=ax); fig.savefig("elem_pos_gram.png", dpi=110)
