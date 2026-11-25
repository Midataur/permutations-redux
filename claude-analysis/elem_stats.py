"""
Test 6a, part 1: the elementary checkpoint on the REAL test file, and hop/width statistics of real training data.
Usage: python elem_stats.py DATA_DIR [N_TRAIN_ROWS] [N_FREE]
Needs train_data1.csv, test_data.csv (+ _perms) from Midataur/permutations @ fdefb8c (see elem_data_stats.py).
"""
import sys, time
import numpy as np
import pandas as pd
from np_transformer import NPTransformer
from elem_tools import *

D = sys.argv[1]
NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
NFREE = int(sys.argv[3]) if len(sys.argv) > 3 else 500
MODEL = "../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
rd = lambda f, n=None: pd.read_csv(f"{D}/{f}", header=None, nrows=n).to_numpy(dtype=np.int64)
t0 = time.time()
Wte, Pte = rd("test_data.csv"), rd("test_data_perms.csv")
Wtr, Ptr = rd("train_data1.csv", NTR), rd("train_data1_perms.csv", NTR)
assert (apply_words_elem(Wte) == Pte).all() and (apply_words_elem(Wtr[:5000]) == Ptr[:5000]).all()
print("stored permutations reproduced by my convention: OK")

# ---- real training data: hops and width per output slot
Hmax = 130
H2 = np.zeros((Hmax, G + 1), np.int64)
for a in range(0, len(Wtr), 10000):
    tr = trajectories_elem(Wtr[a:a + 10000])
    np.add.at(H2, (np.minimum(tr["hops"].ravel(), Hmax - 1), tr["width"].ravel()), 1)
slots = H2.sum()
print(f"\nreal TRAINING data ({len(Wtr)} words, {slots} slots): per-slot hops and width")
hops_marg = H2.sum(1); width_marg = H2.sum(0)
cumL = lambda L: hops_marg[L:].sum() / slots
print("  hops: mean %.1f; quantiles 5/50/95/99/99.9%%:" % ((hops_marg * np.arange(Hmax)).sum() / slots),
      [int(np.searchsorted(np.cumsum(hops_marg) / slots, q)) for q in (.05, .5, .95, .99, .999)])
print("  P(slot has >= L hops):", {L: float(f"{cumL(L):.4g}") for L in (10, 20, 30, 40, 50, 60, 80, 100)})
print("  width (distinct positions visited) distribution:", {w: int(c) for w, c in enumerate(width_marg) if c})
print("  P(width >= 12):", float(width_marg[12:].sum() / slots), "   (training windows give widths <= 11 in a single window)")
np.savez("elem_train_chase_hist.npz", H2=H2)

# ---- the model on the real test file
m = NPTransformer(MODEL, max_group=G, input_len=N, num_trans=G)
tr_te = trajectories_elem(Wte, check=True)
print(f"\nreal TEST file ({len(Wte)} words): hops mean {tr_te['hops'].mean():.1f}, max {tr_te['hops'].max()}; "
      f"width mean {tr_te['width'].mean():.2f}, max {tr_te['width'].max()}; P(width>=12) = {np.mean(tr_te['width'] >= 12):.4f}")
_, correct, margin = tf_eval_elem(m, Wte, Pte)
print(f"  teacher-forced: slot accuracy {correct.mean():.5f}, word accuracy {correct.all(1).mean():.4f}  [{time.time()-t0:.0f}s]")
fr = np.zeros(NFREE, bool)
for a in range(0, NFREE, 50):
    g = m.generate(Wte[a:a + 50], force_valid=True)
    fr[a:a + 50] = (g == Pte[a:a + 50]).all(1)
print(f"  free-running exact match on the first {NFREE} test words: {fr.mean():.4f} ({fr.sum()}/{NFREE})  [{time.time()-t0:.0f}s]")
bad = np.argwhere(~correct)
print("  teacher-forced errors (word, slot, hops, width):", [(int(i), int(k), int(tr_te['hops'][i, k]), int(tr_te['width'][i, k])) for i, k in bad[:15]])
