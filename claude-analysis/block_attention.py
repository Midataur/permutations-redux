"""
Does the network use the position block boundary?  For elementary models with and without a block boundary in their
position embeddings (block_survey.py), measure whether attention avoids crossing the boundary.

On N natural test words, for each layer and head, attention between word positions i and j with 1 <= |i - j| <= D
(D = 6): for every candidate boundary c in 10..110, the mean attention weight of the pairs that straddle c
(i < c <= j or j < c <= i), divided by the mean over all c.  A dip at the model's boundary b means that the heads
attend less across b than across other places at the same distance.  Also: at the 16 query positions, the mean
attention to each word position (to see whether heads read ranges that end at b).

Usage: python block_attention.py TEST_CSV N name=path:blocks:boundary ...
Writes block_attention.json and fig_block_attention.png.
"""
import sys, json
import numpy as np
import np_transformer
from np_transformer import NPTransformer
from elem_tools import G, N as NW, apply_words_elem

_orig = np_transformer.load_safetensors
np_transformer.load_safetensors = lambda p: ({k.replace("_orig_mod.", ""): v for k, v in _orig(p)[0].items()}, None)

test_csv, NWORDS = sys.argv[1], int(sys.argv[2])
specs = [a.split("=", 1) for a in sys.argv[3:]]
words = np.loadtxt(test_csv, delimiter=",", dtype=np.int64)[:NWORDS]
perms = apply_words_elem(words)
D = 6
res = {}
for name, rest in specs:
    path, blocks, b = rest.rsplit(":", 2)
    m = NPTransformer(path, max_group=G, input_len=NW, num_trans=G, n_blocks=int(blocks))
    start = np.full((len(words), 1), m.START)
    toks = np.concatenate([words, start, perms[:, :-1] + m.num_trans], axis=1)
    A = None
    for s in range(0, len(words), 50):
        _, at = m.forward(toks[s:s + 50], return_attn=True)
        a = np.stack([x.mean(0) for x in at])          # (layers, heads, T, T), mean over words
        A = a * len(toks[s:s + 50]) if A is None else A + a * len(toks[s:s + 50])
    A /= len(words)
    Lw = NW
    i, j = np.indices((Lw, Lw)); near = (np.abs(i - j) >= 1) & (np.abs(i - j) <= D)
    cs = np.arange(10, 111)
    prof = np.zeros(A.shape[:2] + (len(cs),))
    for li in range(A.shape[0]):
        for h in range(A.shape[1]):
            W = A[li, h, :Lw, :Lw]
            for ci, c in enumerate(cs):
                cross = near & (((i < c) & (j >= c)) | ((j < c) & (i >= c)))
                prof[li, h, ci] = W[cross].mean()
            prof[li, h] /= prof[li, h].mean()
    q = A[:, :, Lw:, :Lw].mean(2)                        # (layers, heads, 120): query positions -> word positions
    b = int(b)
    at_b = prof[:, :, list(cs).index(b)] if b in cs else None
    res[name] = dict(boundary=b, cross_ratio_at_b=None if at_b is None else np.round(at_b, 3).tolist(),
                     min_ratio_heads=[(int(li + 1), int(h), int(cs[prof[li, h].argmin()]), round(float(prof[li, h].min()), 3))
                                      for li in range(prof.shape[0]) for h in range(prof.shape[1]) if prof[li, h].min() < 0.7],
                     profile=prof.tolist(), query=q.tolist())
    print(f"{name}: boundary {b}; heads whose cross-attention dips below 0.7 of their mean (block, head, where, ratio):",
          res[name]["min_ratio_heads"], flush=True)
json.dump(res, open("block_attention.json", "w"))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, len(res), figsize=(3.2 * len(res), 5.6), squeeze=False)
for k, (name, r) in enumerate(res.items()):
    prof = np.array(r["profile"]); q = np.array(r["query"])
    ax = axs[0, k]
    for li in range(prof.shape[0]):
        for h in range(prof.shape[1]):
            ax.plot(cs, prof[li, h], lw=0.6, color=plt.cm.viridis(li / max(1, prof.shape[0] - 1)), alpha=0.8)
    ax.axvline(r["boundary"], color="r", lw=0.8, ls=":"); ax.set_ylim(0, 2); ax.set_title(name, fontsize=8)
    ax.set_xlabel("place c", fontsize=7); ax.set_ylabel("attention across c, |i-j|<=6 (/ mean)", fontsize=6); ax.tick_params(labelsize=6)
    ax = axs[1, k]
    for li in range(q.shape[0]):
        for h in range(q.shape[1]):
            ax.plot(q[li, h], lw=0.6, color=plt.cm.viridis(li / max(1, q.shape[0] - 1)), alpha=0.8)
    ax.axvline(r["boundary"], color="r", lw=0.8, ls=":"); ax.set_xlabel("word position", fontsize=7)
    ax.set_ylabel("attention from the 16 query positions", fontsize=6); ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig("fig_block_attention.png", dpi=130)
print("wrote fig_block_attention.png")
