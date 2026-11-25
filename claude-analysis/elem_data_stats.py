"""
Is there any structure at word position 45 in the REAL training data of the elementary model
(elem-partitioned-long-boosted-1; Midataur/permutations @ fdefb8c, scaling-generator/data/elem_partitioned_long_boosted/)?

Checks, on train_data1.csv (+ optionally train_data2.csv) and the test file:
  0. shapes, token histogram, identity share
  1. which token -> adjacent-transposition convention reproduces the stored permutations
  2. per-position token histogram: is the data position-stationary? (chi-square per position, extremes, window 40-50)
  3. prefix statistics as a function of position i: Coxeter length (inversions) of the product of the first i letters,
     number of distinct tokens seen so far, number of non-identity letters so far
  4. row structure along the file (windows used per row; is the file ordered?)

Usage: python elem_data_stats.py DATA_DIR [N_FILES]
"""
import sys, itertools
import numpy as np
import pandas as pd

D = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rd = lambda f: pd.read_csv(f"{D}/{f}", header=None).to_numpy(dtype=np.int64)

W = np.concatenate([rd(f"train_data{k}.csv") for k in range(1, NF + 1)])
P = np.concatenate([rd(f"train_data{k}_perms.csv") for k in range(1, NF + 1)])
Wt, Pt = rd("test_data.csv"), rd("test_data_perms.csv")
N, T = W.shape
G = P.shape[1]
print(f"train: words {W.shape}, perms {P.shape}; test: words {Wt.shape}, perms {Pt.shape}")
print(f"token range train {W.min()}..{W.max()}, test {Wt.min()}..{Wt.max()}")
tok_tr = np.bincount(W.ravel(), minlength=G) / W.size
tok_te = np.bincount(Wt.ravel(), minlength=G) / Wt.size
print("token frequencies  train:", np.round(tok_tr, 4))
print("token frequencies  test :", np.round(tok_te, 4))


# ---------------------------------------------------------------- 1. convention
def make_letters(conv):
    """Return list L[token] = (a, b) positions to swap, or None for identity."""
    L = []
    for t in range(G):
        a = t + conv["off"]
        L.append((a, a + 1) if 0 <= a and a + 1 < G else None)
    return L


def apply(words, L, mode):
    B = len(words)
    arr = np.tile(np.arange(G), (B, 1))
    rows = np.arange(B)
    for i in range(words.shape[1]):
        t = words[:, i]
        for tok, sw in enumerate(L):
            if sw is None:
                continue
            m = t == tok
            if not m.any():
                continue
            a, b = sw
            if mode == "positions":          # swap the entries at positions a, b
                x, y = arr[m, a].copy(), arr[m, b].copy()
                arr[m, a], arr[m, b] = y, x
            else:                            # swap the entries whose VALUES are a, b
                ia = np.argmax(arr[m] == a, 1); ib = np.argmax(arr[m] == b, 1)
                r = np.nonzero(m)[0]
                arr[r, ia], arr[r, ib] = b, a
    return arr


best = None
print("\n1. convention search on the first 3000 training rows (fraction of rows whose stored perm is reproduced):")
for off, mode in itertools.product((-1, 0), ("positions", "values")):
    L = make_letters(dict(off=off))
    got = apply(W[:3000], L, mode)
    frac = float((got == P[:3000]).all(1).mean())
    print(f"   token t swaps ({'t+%d' % off}, {'t+%d' % (off + 1)}), mode={mode:9s}: {frac:.4f}")
    if best is None or frac > best[0]:
        best = (frac, off, mode)
frac, off, mode = best
print(f"   -> using off={off}, mode={mode} (match {frac:.4f})")
L = make_letters(dict(off=off))
ident_tokens = [t for t, s in enumerate(L) if s is None]
print("   identity tokens under this convention:", ident_tokens)

# ---------------------------------------------------------------- 2. position stationarity
print("\n2. per-position token histogram (stationarity)")


def pos_hist(Wm):
    H = np.zeros((Wm.shape[1], G), np.int64)
    for t in range(G):
        H[:, t] = (Wm == t).sum(0)
    return H


def chi2_per_position(Wm):
    H = pos_hist(Wm).astype(float)
    pooled = H.sum(0) / H.sum()
    exp = pooled[None] * Wm.shape[0]
    ok = exp[0] > 5
    chi = ((H[:, ok] - exp[:, ok]) ** 2 / exp[:, ok]).sum(1)
    dof = ok.sum() - 1
    z = (chi - dof) / np.sqrt(2 * dof)
    return chi, dof, z


for name, Wm in (("train", W), ("test", Wt)):
    chi, dof, z = chi2_per_position(Wm)
    print(f"   {name}: chi-square per position (dof {dof}): mean {chi.mean():.1f}, "
          f"z-scores min {z.min():.2f} max {z.max():.2f}; positions with |z|>3: {list(np.nonzero(np.abs(z) > 3)[0])}")
    print(f"      z at positions 40..50: {np.round(z[40:51], 2)}")
    # slow trend: correlation of each token's frequency with position
    H = pos_hist(Wm) / Wm.shape[0]
    slopes = [np.polyfit(np.arange(T), H[:, t], 1)[0] * T for t in range(G)]
    print(f"      total drift over the word, per token (frequency change first->last position): {np.round(slopes, 4)}")

# ---------------------------------------------------------------- 3. prefix statistics
print("\n3. prefix statistics by position (mean over rows)")


def prefix_stats(Wm, nrows=20000):
    Wm = Wm[:nrows]
    B = len(Wm)
    arr = np.tile(np.arange(G), (B, 1))
    length = np.zeros(T + 1)
    seen = np.zeros((B, G), bool)
    distinct = np.zeros(T + 1)
    nonid = np.zeros(T + 1)
    cnt = np.zeros(B)
    inv_of = lambda a: (a[:, :, None] > a[:, None, :]).triu(1).sum((1, 2)) if hasattr(a, "triu") else \
        np.triu(a[:, :, None] > a[:, None, :], 1).sum((1, 2))
    length[0] = 0
    for i in range(T):
        tcol = Wm[:, i]
        for tok, sw in enumerate(L):
            if sw is None:
                continue
            m = tcol == tok
            a, b = sw
            if mode == "positions":
                x, y = arr[m, a].copy(), arr[m, b].copy(); arr[m, a], arr[m, b] = y, x
            else:
                ia = np.argmax(arr[m] == a, 1); ib = np.argmax(arr[m] == b, 1)
                r = np.nonzero(m)[0]; arr[r, ia], arr[r, ib] = b, a
        seen[np.arange(B), tcol] = True
        cnt += np.array([L[t] is not None for t in range(G)])[tcol]
        length[i + 1] = np.triu(arr[:, :, None] > arr[:, None, :], 1).sum((1, 2)).mean()
        distinct[i + 1] = seen.sum(1).mean()
        nonid[i + 1] = cnt.mean()
    return length, distinct, nonid


for name, Wm in (("train", W), ("test", Wt)):
    ln, ds, ni = prefix_stats(Wm, 20000 if name == "train" else len(Wm))
    d2 = np.diff(ln, 2)
    print(f"   {name}: Coxeter length after i letters, i = 0,10,20,30,40,45,50,60,80,100,120:")
    print("      ", {i: round(float(ln[i]), 2) for i in (0, 10, 20, 30, 40, 45, 50, 60, 80, 100, 120)})
    print(f"      distinct tokens seen after i letters (10,20,45,120): {[round(float(ds[i]), 2) for i in (10, 20, 45, 120)]}")
    print(f"      largest |second difference| of mean length: at i={int(np.argmax(np.abs(d2))) + 1} ({np.abs(d2).max():.4f}); "
          f"value near 45: {np.round(d2[42:47], 4)}")
    print(f"      mean length at i=45: {ln[45]:.2f}; max length of the group touched: ")

# ---------------------------------------------------------------- 4. row structure
print("\n4. row structure (is the file ordered / which windows are used?)")
lo = W.min(1); hi = W.max(1); nd = np.array([len(set(r)) for r in W[:50000]])
span = hi - lo + 1
print("   window span (max-min+1 of tokens used) over rows: ", dict(zip(*np.unique(span, return_counts=True))))
print("   number of distinct tokens per row (first 50000):  ", dict(zip(*np.unique(nd, return_counts=True))))
runs = np.mean(np.array([span[i] == span[i + 1] for i in range(len(span) - 1)]))
print(f"   P(consecutive rows have the same span) = {runs:.3f}  (iid rows would give {np.sum((np.bincount(span) / len(span)) ** 2):.3f})")
print("   first 12 rows: (min token, max token):", [(int(a), int(b)) for a, b in zip(lo[:12], hi[:12])])
mv = (P != np.arange(G)[None]).sum(1)
print("   moved points of the output perm: ", dict(zip(*np.unique(mv, return_counts=True))))

# ---------------------------------------------------------------- 5. outputs, and file-to-file consistency
print("\n5. stored permutations and consistency between files")
inv = lambda A: np.triu(A[:, :, None] > A[:, None, :], 1).sum((1, 2))
for name, PP in (("train", P[:100000]), ("test", Pt)):
    c = inv(PP)
    print(f"   {name}: Coxeter length of the stored output perm: mean {c.mean():.2f}, max {c.max()}, "
          f"P(length >= 45) = {np.mean(c >= 45):.5f}; moved points: mean {(PP != np.arange(G)).sum(1).mean():.2f}, max {(PP != np.arange(G)).sum(1).max()}")
if NF >= 2:
    n1 = len(W) // NF
    h1 = np.bincount(W[:n1].ravel(), minlength=G) / W[:n1].size
    h2 = np.bincount(W[n1:].ravel(), minlength=G) / W[n1:].size
    print(f"   token histograms of file 1 vs file 2: max abs difference {np.abs(h1 - h2).max():.5f}")
print("   full window-span histogram:", dict(zip(*np.unique(span, return_counts=True))))
print("   full moved-points histogram of train outputs:", dict(zip(*np.unique(mv, return_counts=True))))
