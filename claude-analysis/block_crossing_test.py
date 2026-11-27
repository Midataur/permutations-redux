"""
Does a chain whose hop letters sit across the position block boundary cost more than one away from it?

Planted elementary chains (L hops, window width w) whose hop letters all lie in one span of S word positions: centred
on the model's boundary b ("across"), or centred well inside the left or the right block ("left", "right").  Teacher-
forced accuracy and median margin at the planted slot.  Models without a boundary are run with b = 72 as a control.

Usage: python block_crossing_test.py N L W S name=path:blocks:b ...   -> block_crossing_results.json
"""
import sys, json
import numpy as np
import np_transformer
from np_transformer import NPTransformer
from elem_tools import G, N, plant_elem_word, tf_eval_elem, trajectories_elem

_orig = np_transformer.load_safetensors
np_transformer.load_safetensors = lambda p: ({k.replace("_orig_mod.", ""): v for k, v in _orig(p)[0].items()}, None)
NB, L, W, S = (int(x) for x in sys.argv[1:5])
specs = [a.split("=", 1) for a in sys.argv[5:]]


def batch(rng, centre):
    lo = max(0, min(N - S, centre - S // 2))
    words, ks = [], []
    while len(words) < NB:
        k = int(rng.integers(0, G))
        pos = np.sort(rng.choice(np.arange(lo, lo + S), size=L, replace=False))   # positions of the hop letters
        reads = np.sort(N - 1 - pos)                                               # reading order (0 = last letter)
        try:
            words.append(plant_elem_word(rng, k, L, W, hop_reads=reads)); ks.append(k)
        except ValueError:
            continue
    words, ks = np.stack(words), np.array(ks)
    tr = trajectories_elem(words, check=True)
    assert (tr["hops"][np.arange(NB), ks] == L).all()
    return words, ks


res = {}
for name, rest in specs:
    path, blocks, b = rest.rsplit(":", 2); b = int(b)
    m = NPTransformer(path, max_group=G, input_len=N, num_trans=G, n_blocks=int(blocks))
    r = {}
    for where, centre in (("left", b // 2), ("across", b), ("right", (b + N) // 2)):
        rng = np.random.default_rng(99)
        w, k = batch(rng, centre)
        _, c, mg = tf_eval_elem(m, w)
        r[where] = dict(centre=centre, acc=float(c[np.arange(NB), k].mean()), margin=float(np.median(mg[np.arange(NB), k])))
    res[name] = r
    print(f"{name:28s} b={b:3d}  " + "  ".join(f"{wh}: {v['acc']:.3f} (margin {v['margin']:.1f})" for wh, v in r.items()), flush=True)
json.dump(dict(N=NB, L=L, W=W, S=S, results=res), open("block_crossing_results.json", "w"), indent=1)
