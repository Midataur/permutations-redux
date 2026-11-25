"""
Which model produced Figure 5b (position-embedding self-similarity, adjacent transpositions, n=16) of arXiv 2502.12717?
Searches the notebook history of Midataur/permutations (scaling-generator/mechanistic.ipynb) and the saved pictures in
scaling-generator/embedding_pictures/, and compares them with the checkpoints in this repo.

  1. decode the saved PNGs (plotly heatmaps) back into similarity values and compare with cos(position_embedding) of the checkpoint
  2. extract every square heatmap in the notebook (plotly JSON in the cell outputs) at a few commits; correlate with the checkpoint
  3. locate the block boundaries of the 136x136 matrix of the elementary checkpoint

Usage: python fig5b_provenance.py PATH_TO_CLONE_OF_Midataur/permutations    (a blob:none clone is enough; blobs are fetched on demand)
"""
import sys, json, subprocess
import numpy as np
import matplotlib.image as mpimg
from st_loader import load_safetensors
from paths import MODEL_TORN
from pathlib import Path

CLONE = sys.argv[1]
MODELS = Path(MODEL_TORN).parent.parent
ELEM = MODELS / "elem-partitioned-long-boosted-1/model.safetensors"
git_blob = lambda spec: subprocess.run(["git", "-C", CLONE, "cat-file", "blob", spec], capture_output=True, check=True).stdout


def cos(P):
    P = P.astype(np.float64); X = P / np.linalg.norm(P, axis=1, keepdims=True); return X @ X.T


C = {"elem-partitioned-long-boosted-1": cos(load_safetensors(str(ELEM))[0]["position_embedding.weight"]),
     "torn-bigger-13": cos(load_safetensors(MODEL_TORN)[0]["position_embedding.weight"])}

# ---- 1. saved pictures. The plot area of the 1000x1000 plotly picture is found from the axis tick positions:
#         136 positions: x 46.7..952.5, y 3.5..909.4;  50 positions: x 45..955, y -1..908. The colour bar (row 968, x 10..989)
#         maps linearly onto [min, max] of the matrix (plotly's autoscale), which gives a colour -> value table.
print("1. saved pictures (commit dcb73d1), decoded and compared with cos(position_embedding) of the checkpoint")
for name, n, box in (("elem-partitioned-long-boosted-1", 136, (46.7, 952.5, 3.5, 909.4)), ("torn-bigger-13", 50, (45, 955, -1, 908))):
    open("/tmp/_pic.png", "wb").write(git_blob(f"dcb73d1:scaling-generator/embedding_pictures/position/{name}.png"))
    im = mpimg.imread("/tmp/_pic.png")[:, :, :3]
    lut = im[968, 10:990, :]; vals = np.linspace(C[name].min(), C[name].max(), len(lut))
    x0, x1, y0, y1 = box; cw, ch = (x1 - x0) / n, (y1 - y0) / n
    dec = np.array([[vals[np.argmin(((lut - im[int(y0 + (i + .5) * ch), int(x0 + (j + .5) * cw)]) ** 2).sum(1))] for j in range(n)] for i in range(n)])
    print(f"   {name}: correlation {np.corrcoef(dec.ravel(), C[name].ravel())[0,1]:.4f}, max |difference| {np.abs(dec - C[name]).max():.3f}")

# ---- 2. notebook outputs
print("\n2. square heatmaps in the notebook outputs, correlation with the checkpoint (all entries / off-band |i-j|>6, inputs only)")
i, j = np.meshgrid(range(136), range(136), indexing="ij"); off = (abs(i - j) > 6) & (i < 120) & (j < 120)
for commit in ("fdefb8c", "5863de5", "f262ca2", "dcb73d1"):
    nb = json.loads(git_blob(f"{commit}:scaling-generator/mechanistic.ipynb"))
    for ci, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        for o in c.get("outputs", []):
            p = o.get("data", {}).get("application/vnd.plotly.v1+json")
            if not p or not p["data"] or p["data"][0].get("type") != "heatmap" or "position_embedding(" not in src: continue
            try: z = np.array(p["data"][0]["z"], dtype=float)
            except Exception: continue
            if z.ndim != 2 or z.shape[0] != z.shape[1]: continue
            ref = C["elem-partitioned-long-boosted-1"] if z.shape[0] == 136 else C["torn-bigger-13"] if z.shape[0] == 50 else None
            if ref is None: print(f"   {commit} cell {ci}: {z.shape}, no matching checkpoint"); continue
            msg = f"{np.corrcoef(z.ravel(), ref.ravel())[0,1]:.3f}" + (f" / {np.corrcoef(z[off], ref[off])[0,1]:.3f}" if z.shape[0] == 136 else "")
            print(f"   {commit} cell {ci:2d} (exec {c.get('execution_count')}): {z.shape[0]}x{z.shape[0]}  {msg}")

# ---- 3. block boundaries of Figure 5b's matrix
G = C["elem-partitioned-long-boosted-1"]
nr = lambda v: (v - v.mean()) / (np.linalg.norm(v - v.mean()) + 1e-9)
print("\n3. block boundaries of the 136x136 matrix (the saved picture = this matrix)")
print("   cosine between neighbouring positions (i, i+1) at the lowest values:",
      [(int(k), round(float(G[k, k + 1]), 3)) for k in sorted(np.argsort([G[k, k + 1] for k in range(135)])[:3])])
sp = [(k + 1, 1 - float(nr(G[k, :53]) @ nr(G[k + 1, :53]))) for k in range(60, 119)]
print("   rows 60-119 compared over columns 0-52 (always off the diagonal): 1 - corr(row k, row k+1); median "
      f"{np.median([s for _, s in sp]):.2f}; largest:", [(k, round(s, 2)) for k, s in sorted(sp, key=lambda t: -t[1])[:3]])
print("   (rows 72 and 73 are transitional; the matrix splits into [0,72) and [72,120) for the word positions, plus the 16 output positions [120,136))")
