"""
Statistics of the REAL training / test files for torn-bigger-13 (commit 5863de5 of Midataur/permutations,
scaling-generator/data/torn_bigger/). Reports, for the words in one training file (400,000 words = 1/20 of the
training set) and for the test file:
  * distinct points touched by the word, points moved by the permutation (training support check)
  * share of identity tokens (a == b)
  * agreement of the stored permutations with the permutation recomputed from the words
  * per-slot pointer-chase hop-count and distinct-point distributions (scaled x20 to the full training set)
and the model's free-running and teacher-forced accuracy on the real test words.

Usage: python data_stats.py DATA_DIR [N_TRAIN_WORDS]
  DATA_DIR contains train_data1.csv, train_data1_perms.csv, test_data.csv, test_data_perms.csv
"""
from paths import MODEL_TORN
import sys
import numpy as np
from permdata import apply_words, trajectories, moved_points
from np_transformer import NPTransformer
from evalutil import tf_eval

D = sys.argv[1]
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
n = 25
load = lambda f, k=None: np.loadtxt(f"{D}/{f}", delimiter=",", dtype=np.int64, max_rows=k)


def support_stats(W, name):
    a, b = W // n, W % n
    N = len(W)
    touched = np.zeros((N, n), bool)
    nz = a != b
    rows = np.broadcast_to(np.arange(N)[:, None], W.shape)
    touched[rows[nz], a[nz]] = True
    touched[rows[nz], b[nz]] = True
    T = touched.sum(1)
    print(f"[{name}] {N} words: identity-token share {np.mean(~nz):.4f}")
    print(f"   touched points: min {T.min()} mean {T.mean():.2f} max {T.max()}   P(>10)={np.mean(T>10):.5f}  P(>16)={np.mean(T>16):.5f}")
    return T


# ------------------------------------------------------------------ training file
Wtr = load("train_data1.csv", NT)
Ptr = load("train_data1_perms.csv", NT)
T = support_stats(Wtr, "train")
print("   histogram of touched points:", dict(zip(*np.unique(T, return_counts=True))))
mine = apply_words(Wtr[:20000])
print("   stored perms == recomputed perms on first 20000 rows:", bool((mine == Ptr[:20000]).all()))
M = moved_points(Ptr)
print(f"   moved points: min {M.min()} mean {M.mean():.2f} max {M.max()}   P(>16)={np.mean(M>16):.5f}")

# per-slot chase statistics, in batches
maxL = 26
hist_L = np.zeros(maxL, np.int64)
hist_s = np.zeros(n + 1, np.int64)
for a0 in range(0, len(Wtr), 20000):
    tr = trajectories(Wtr[a0:a0 + 20000], check=False)
    hist_L += np.bincount(tr["hops"].ravel(), minlength=maxL)[:maxL]
    hist_s += np.bincount(tr["distinct"].ravel(), minlength=n + 1)[:n + 1]
scale = 8_000_000 / len(Wtr)
print(f"\nper-slot chase statistics, scaled to the full 8M-word training set (x{scale:.0f}):")
print("   P(slot has >= L hops) -> expected count in 8M words")
for L in (8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20):
    c = hist_L[L:].sum()
    print(f"   L>={L:2d}: observed in file {c:9d}   -> ~{c * scale:12.1f} in 8M words")
print("   distinct points visited by a slot's chase (observed in file):", {s: int(c) for s, c in enumerate(hist_s) if c})
np.savez("train_chase_hist.npz", hist_L=hist_L, hist_s=hist_s, scale=scale)

# ------------------------------------------------------------------ real test file
Wte = load("test_data.csv")
Pte = load("test_data_perms.csv")
support_stats(Wte, "test")
print("   stored perms == recomputed perms (test):", bool((apply_words(Wte) == Pte).all()))
tr = trajectories(Wte, check=False)
print("   test: max hops per slot", tr["hops"].max(), " max distinct", tr["distinct"].max())

m = NPTransformer(MODEL_TORN)
fr = np.zeros(len(Wte), bool)
for a0 in range(0, len(Wte), 100):
    g = m.generate(Wte[a0:a0 + 100], force_valid=True)
    fr[a0:a0 + 100] = (g == Pte[a0:a0 + 100]).all(1)
_, correct, _, _ = tf_eval(m, Wte)
print(f"\nmodel on the real test file ({len(Wte)} words): free-running exact match {fr.mean():.4f} "
      f"({fr.sum()}/{len(Wte)}); teacher-forced word acc {correct.all(1).mean():.4f}; slot acc {correct.mean():.5f}")
