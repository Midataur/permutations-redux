"""Check the GPU data generators against NumPy references and against the statistics measured on the REAL training data
(claude-analysis/*.txt|npz).  Runs on CPU.   python tests/test_datagen.py"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); AN = os.path.join(ROOT, "..", "claude-analysis")
sys.path.insert(0, ROOT)
from exp import np_tools as T
from exp.datagen import GeneralSampler, ElemSampler, apply_swaps_general, apply_swaps_elem, general_batch, elem_batch, ELEM_PARTITIONS

print("partitions of 10 into parts >= 3:", ELEM_PARTITIONS)
assert ELEM_PARTITIONS == [(10,), (7, 3), (6, 4), (5, 5), (4, 3, 3)]

# 1. permutation routines agree with the NumPy ones (which the earlier analysis checked against the real stored permutations)
rng = np.random.default_rng(0)
w = rng.integers(0, 625, (500, 25))
assert np.array_equal(apply_swaps_general(torch.as_tensor(w)).numpy(), T.apply_words(w)); print("general permutation: OK")
we = rng.integers(0, 16, (500, 120))
assert np.array_equal(apply_swaps_elem(torch.as_tensor(we)).numpy(), T.apply_words_elem(we)); print("elementary permutation: OK")

# 2. batch layout
S = GeneralSampler("cpu", dict(ms={10: 1.0}), 1)
words, perm = S.sample(4); inp, tgt = general_batch(words, perm)
assert inp.shape == (4, 50) and tgt.shape == (4, 25) and (inp[:, 25] == 651).all() and (inp[:, 26:] == perm[:, :-1] + 625).all() and (tgt == perm + 625).all()
E = ElemSampler("cpu", 1); we_, pe = E.sample(4); ie, te = elem_batch(we_, pe)
assert ie.shape == (4, 136) and te.shape == (4, 16) and (ie[:, 120] == 33).all() and (ie[:, 121:] == pe[:, :-1] + 16).all()
print("batch layout: OK")

# 3. natural m=10 words: the per-slot hop histogram matches the REAL training data (hist_L in train_chase_hist.npz)
ref = np.load(os.path.join(AN, "train_chase_hist.npz")); hL = ref["hist_L"].astype(float); hL /= hL.sum()
S = GeneralSampler("cpu", dict(ms={10: 1.0}), 2)
big = np.concatenate([S.natural(2000).numpy() for _ in range(25)])             # 50k words
tr = T.trajectories(big, check=False); mine = np.bincount(tr["hops"].ravel(), minlength=26)[:26].astype(float); mine /= mine.sum()
print("slots with L hops:  L: real   mine")
for L in range(0, 14):
    print(f"   {L:2d}  {hL[L]:.5f}  {mine[L]:.5f}")
assert np.abs(hL[:12] - mine[:12]).max() < 0.004, "natural generator differs from the real data"
d = np.bincount(tr["distinct"].ravel(), minlength=26)[:26].astype(float); d /= d.sum(); hs = ref["hist_s"].astype(float); hs /= hs.sum()
assert np.abs(hs[:11] - d[:11]).max() < 0.004
words_touch = np.array([len(set(((x // 25).tolist()) + ((x % 25).tolist()))) for x in big[:5000]])
print("points touched per word (mine): mean %.2f max %d   (real: mean 9.93, max 10)" % (words_touch.mean(), words_touch.max()))
assert words_touch.max() <= 10 and abs(words_touch.mean() - 9.93) < 0.1
print("natural general generator matches the real training data: OK")

# 4. planted words: exactly L hops through exactly s points at the planted slot; the whole word stays on m points
for m, Lr in ((10, (3, 25)), (5, (3, 25))):
    S = GeneralSampler("cpu", dict(ms={m: 1.0}), 3)
    wd, kk, L, s = S.planted(3000, m, *Lr)
    wd = wd.numpy(); kk, L, s = kk.numpy(), L.numpy(), s.numpy()
    tr = T.trajectories(wd)                                                      # also asserts pointer picture == permutation
    idx = np.arange(len(wd))
    assert (tr["hops"][idx, kk] == L).all() and (tr["distinct"][idx, kk] == s).all(), "planted chain has wrong L or s"
    touched = np.array([len(set(((x // 25).tolist()) + ((x % 25).tolist()))) for x in wd])
    assert touched.max() <= m
    assert L.min() >= Lr[0] and L.max() == Lr[1] and s.min() >= 2 and s.max() <= m and (s <= L + 1).all()
    print(f"planted m={m}: exact (L,s) at the planted slot for 3000/3000 words; L in [{L.min()},{L.max()}], s in [{s.min()},{s.max()}], points touched <= {touched.max()}")
# mixture batch
S = GeneralSampler("cpu", dict(ms={10: 1.0}, planted_frac=0.5, plant_m=10, Lmin=3, Lmax=25), 4)
wd, pm = S.sample(2000); tr = T.trajectories(wd.numpy())
assert np.array_equal(pm.numpy(), T.apply_words(wd.numpy()))
share = (tr["hops"] >= 15).mean(); print("mixture arm: share of slots with >=15 hops: %.4f, >=20 hops: %.4f" % (share, (tr["hops"] >= 20).mean()))

# 5. elementary generator vs the REAL data: token frequencies, window span, distinct tokens, (hops, width) histogram
E = ElemSampler("cpu", 5)
we_ = E.words(20000).numpy()
freq = np.bincount(we_.ravel(), minlength=16) / we_.size
real = np.array([0.0205, 0.0394, 0.057, 0.0686, 0.0765, 0.0814, 0.085, 0.0854, 0.0854, 0.0854, 0.0805, 0.0736, 0.0637, 0.0508, 0.0313, 0.0156])
print("token frequencies real:", real.round(3)); print("token frequencies mine:", freq.round(3))
assert np.abs(freq - real).max() < 0.003
span = we_.max(1) - we_.min(1) + 1
real_span = {4: 1615, 5: 10478, 6: 36220, 7: 57632, 8: 38151, 9: 35339, 10: 112379, 11: 29193, 12: 25172, 13: 20882, 14: 16080, 15: 11148, 16: 5711}
tot = sum(real_span.values()); mspan = np.bincount(span, minlength=17)
print("window span  real%  mine%:")
for k in range(4, 17):
    print(f"   {k:2d}  {100*real_span[k]/tot:5.2f} {100*mspan[k]/len(span):5.2f}")
assert max(abs(real_span[k] / tot - mspan[k] / len(span)) for k in range(4, 17)) < 0.01
H2 = np.load(os.path.join(AN, "elem_train_chase_hist.npz"))["H2"].astype(float); H2 /= H2.sum()
big = np.concatenate([E.words(1000).numpy() for _ in range(20)])
tr = T.trajectories_elem(big)
M = np.zeros_like(H2); np.add.at(M, (np.minimum(tr["hops"].ravel(), 129), np.minimum(tr["width"].ravel(), 16)), 1); M /= M.sum()
print("(hops, width) histogram: total variation distance real vs mine = %.4f" % (0.5 * np.abs(H2 - M).sum()))
print("share of slots with >=40 hops: real %.4f mine %.4f ; >=56 hops: real %.5f mine %.5f" % (H2[40:].sum(), M[40:].sum(), H2[56:].sum(), M[56:].sum()))
print("max hops seen: real", int(np.nonzero(H2.sum(1))[0].max()), "mine", int(tr["hops"].max()))
assert 0.5 * np.abs(H2 - M).sum() < 0.03
print("ALL DATAGEN TESTS PASSED")
