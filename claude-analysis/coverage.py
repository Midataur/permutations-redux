"""
The coverage half of the density argument (report section 4.2, Step 2), written out and checked.

Objects: a slot k of a word, through its chain: the hop letters (with their positions) and the walk of points the
pointer visits; the other letters only have to avoid the pointer.  Both distributions are invariant under relabelling
the 25 points, so it is enough to compare one labelled chain.

  test  (the paper's test words): 25 letters, each uniform over the 625 tokens.
  train (the paper's training words): a uniformly random set M of 10 of the 25 points, then 25 letters each uniform over
        the 100 tokens on M (natural 10-point words relabelled into 25).

For a fixed labelled chain with L hops visiting a set V of s points (the slot's own point included), at fixed letter
positions:
  P_test  = (2/625)^L * (577/625)^(25-L)        (each hop letter is one of the two tokens {a,b}; every other letter
                                                 avoids the pointer's point: 625 - 48 tokens)
  P_train = P(V within M) * (2/100)^L * (82/100)^(25-L),  P(V within M) = C(25-s, 10-s) / C(25, 10)
so the ratio depends only on (L, s):
  R(L, s) = P_train / P_test = C(25-s,10-s)/C(25,10) * (625/100)^L * (82/100 * 625/577)^(25-L)      for s <= 10,
  and P_train = 0 for s >= 11.
Since the ratio is the same for every chain in a class (L, s), it is also the ratio of the class probabilities, which
this script checks by simulation (N words from each distribution), and it evaluates the classes of the real test file.

Usage: python coverage.py N_WORDS TEST_CSV   -> coverage_results.json
"""
import sys, json
from math import comb
import numpy as np
from permdata import trajectories

N = int(sys.argv[1]); test_csv = sys.argv[2]
n = 25
rng = np.random.default_rng(424242)


def R(L, s):
    if s > 10:
        return 0.0
    return comb(25 - s, 10 - s) / comb(25, 10) * (625 / 100) ** L * (0.82 * 625 / 577) ** (25 - L)


def classes(words):
    cnt = {}
    for s0 in range(0, len(words), 100_000):
        tr = trajectories(words[s0:s0 + 100_000], check=False)
        L, S = tr["hops"].ravel(), tr["distinct"].ravel()
        u, c = np.unique(L * 100 + S, return_counts=True)
        for k, v in zip(u, c):
            cnt[int(k)] = cnt.get(int(k), 0) + int(v)
    return cnt


test_w = rng.integers(0, 625, size=(N, 25))
a = rng.integers(0, 10, (N, 25)); b = rng.integers(0, 10, (N, 25)); rel = rng.random((N, n)).argsort(1)
train_w = np.take_along_axis(rel, a, 1) * n + np.take_along_axis(rel, b, 1)
ct, cr = classes(test_w), classes(train_w)
tot = N * 25
rows = []
for key in sorted(set(ct) | set(cr)):
    L, s = divmod(key, 100)
    pt, pr = ct.get(key, 0) / tot, cr.get(key, 0) / tot
    rows.append(dict(L=L, s=s, p_test=pt, p_train=pr, n_test=ct.get(key, 0), n_train=cr.get(key, 0),
                     ratio_sim=(pr / pt if pt > 0 else None), ratio_formula=R(L, s)))
print(f"{'L':>3} {'s':>3} {'p_test':>10} {'p_train':>10} {'ratio sim':>10} {'formula':>10}  (classes with >= 200 test slots)")
for r in rows:
    if r["n_test"] >= 200 and r["L"] > 0:
        print(f"{r['L']:3d} {r['s']:3d} {r['p_test']:10.3e} {r['p_train']:10.3e} {r['ratio_sim'] if r['ratio_sim'] is not None else float('nan'):10.3f} {r['ratio_formula']:10.3f}")
unc = sum(r["p_test"] for r in rows if r["s"] >= 11)
print(f"test slots whose chain visits 11 or more points (not in the training support): {unc:.2e} per slot "
      f"({sum(r['n_test'] for r in rows if r['s'] >= 11)} of {tot:,})")

# the real test file
W = np.loadtxt(test_csv, delimiter=",", dtype=np.int64)
tr = trajectories(W, check=False)
L, S = tr["hops"].ravel(), tr["distinct"].ravel()
moved = L > 0
rs = np.array([R(int(l), int(s)) for l, s in zip(L[moved], S[moved])])
pr_train = {}
for r in rows:
    pr_train[(r["L"], r["s"])] = r["p_train"]
cls = sorted({(int(l), int(s)) for l, s in zip(L[moved], S[moved])})
real = dict(words=len(W), slots=int(L.size), max_hops=int(L.max()), max_points=int(S.max()),
            min_ratio=float(rs.min()), median_ratio=float(np.median(rs)),
            classes=[dict(L=l, s=s, slots=int(((L == l) & (S == s)).sum()), ratio=R(l, s),
                          expected_in_8M_training_words=R(l, s) * 0 + pr_train.get((l, s), 0.0) * 8e6 * 25) for l, s in cls])
print(f"real test file: {len(W)} words; longest chain {real['max_hops']} hops, most points {real['max_points']}; "
      f"ratio train/test over its moved slots: min {real['min_ratio']:.3f}, median {real['median_ratio']:.2f}")
worst = sorted(real["classes"], key=lambda c: c["expected_in_8M_training_words"])[:5]
for c in worst:
    print(f"   rarest classes in training: L={c['L']}, s={c['s']}: {c['slots']} test slots; expected training slots in 8M words "
          f"{c['expected_in_8M_training_words']:.0f}; ratio {c['ratio']:.3f}")
json.dump(dict(rows=rows, uncovered_per_slot=unc, real_test_file=real), open("coverage_results.json", "w"), indent=1)
