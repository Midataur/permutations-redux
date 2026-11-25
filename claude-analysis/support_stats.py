"""
How far outside a 16-point training support are the paper-style test words?

A training word (config: ACTUAL_GROUP_SIZE = 16, RELABEL = True) only ever touches <= 16 points. For fresh uniform
words of 25 general-transposition letters on 25 points we count
  touched : distinct points appearing in any non-identity letter (the word's support)
  moved   : points not fixed by the resulting permutation
and report how often each exceeds 16 (and 10, the paper's value).

Usage: python support_stats.py [N_WORDS]     (default 200000; a few seconds)
"""
import sys
import numpy as np
from permdata import random_words, apply_words, moved_points

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
n = 25
rng = np.random.default_rng(2027)
W = random_words(rng, N)
a, b = W // n, W % n
touched = np.zeros((N, n), bool)
nz = a != b
rows = np.broadcast_to(np.arange(N)[:, None], W.shape)
touched[rows[nz], a[nz]] = True
touched[rows[nz], b[nz]] = True
T = touched.sum(1)
M = moved_points(apply_words(W))
for name, x in (("touched points", T), ("moved points", M)):
    print(f"{name:15s} min {x.min():2d}  mean {x.mean():5.2f}  max {x.max():2d}   "
          f"P(>16) = {np.mean(x > 16):.6f}   P(>10) = {np.mean(x > 10):.6f}")
