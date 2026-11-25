"""Fixed evaluation sets (identical for every arm of the same problem type), built once on the CPU with NumPy.

General (S_25) problem:
  nat25       natural words over all 25 points, like the paper's test file (every word touches ~22 points)
  nat10       natural words over 10 points  = the ORIGINAL training distribution
  ladder_full planted chains (L hops, s distinct points), the other letters over ALL 25 points   (out of support)
  ladder_in10 planted chains whose whole word lives on 10 points                                  (in the m=10 support)
  ladder_in5  planted chains whose whole word lives on 5 points                                   (in the m=5 support)
Elementary problem:
  nat16       uniform tokens (like the paper's test file)
  train_like  the partitioned-window training distribution
  cells       planted chains: width w x hop count L
"""
import numpy as np
from . import np_tools as T

L_GRID = [4, 8, 12, 14, 16, 18, 20, 22, 24]
S_GRID = [3, 6, 10, 14, 20]


def _natural(rng, name, words, n=25, elem=False):
    if elem:
        perm = T.apply_words_elem(words); tr = T.trajectories_elem(words)
        return {f"{name}__words": words, f"{name}__perm": perm, f"{name}__hops": tr["hops"], f"{name}__width": tr["width"]}
    perm = T.apply_words(words, n); tr = T.trajectories(words, n, check=False)
    return {f"{name}__words": words, f"{name}__perm": perm, f"{name}__hops": tr["hops"], f"{name}__distinct": tr["distinct"]}


def build_general(seed=1234, n_nat=2000, per_cell=300):
    rng = np.random.default_rng(seed)
    out = {}
    out.update(_natural(rng, "nat25", T.natural_words(rng, n_nat, 25)))
    out.update(_natural(rng, "nat10", T.natural_words(rng, n_nat, 10)))
    for name, support, s_list in (("ladder_full", None, S_GRID), ("ladder_in10", 10, [3, 6, 10]), ("ladder_in5", 5, [3, 5])):
        W, K, LL, SS = [], [], [], []
        for L in L_GRID:
            for s in s_list:
                if s > L + 1:
                    continue
                w, k = T.plant_batch(rng, per_cell, L, s, support=support)
                W.append(w); K.append(k); LL.append(np.full(per_cell, L)); SS.append(np.full(per_cell, s))
        words = np.concatenate(W)
        out[f"{name}__words"] = words
        out[f"{name}__perm"] = T.apply_words(words)
        out[f"{name}__k"] = np.concatenate(K)
        out[f"{name}__L"] = np.concatenate(LL)
        out[f"{name}__s"] = np.concatenate(SS)
    return out


def build_elem(seed=4321, n_nat=2000, per_cell=300, train_like_words=None):
    rng = np.random.default_rng(seed)
    out = {}
    out.update(_natural(rng, "nat16", rng.integers(0, 16, (n_nat, 120)), elem=True))
    if train_like_words is not None:
        out.update(_natural(rng, "train_like", train_like_words, elem=True))
    W, K, LL, WW = [], [], [], []
    for L in [20, 32, 48, 64, 80]:
        for w in [5, 8, 11, 13, 16]:
            words, ks = T.plant_elem_batch(rng, per_cell, L, w)
            W.append(words); K.append(ks); LL.append(np.full(per_cell, L)); WW.append(np.full(per_cell, w))
    words = np.concatenate(W)
    out["cells__words"] = words
    out["cells__perm"] = T.apply_words_elem(words)
    out["cells__k"] = np.concatenate(K); out["cells__L"] = np.concatenate(LL); out["cells__w"] = np.concatenate(WW)
    return out


def set_names(d):
    return sorted({k.split("__")[0] for k in d})
