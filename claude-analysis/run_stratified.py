"""
Test A (baseline + stratified accuracy). Random test words mirroring the Rust generator
(general transpositions, n=25, N=25 letters, letters iid uniform on n^2 tokens).

Stage 1: free-running greedy decoding with force_valid (the repo's own accuracy metric) on the first N_FREE words.
Stage 2: teacher-forced argmax per output position on N_TF words (gives per-position correctness),
         together with each position's pointer-trajectory statistics (hops, distinct points).
"""
from paths import MODEL_TORN
import sys, time
import numpy as np
from np_transformer import NPTransformer
from permdata import *

N_FREE, N_TF, BS = int(sys.argv[1]), int(sys.argv[2]), 100
MODEL = MODEL_TORN
m = NPTransformer(MODEL)
rng = np.random.default_rng(12345)
words = random_words(rng, N_TF)
perms = apply_words(words)
tr = trajectories(words)
moved = moved_points(perms)

# ---- stage 1: free-running (repo metric)
t0 = time.time()
free_ok = np.zeros(N_FREE, dtype=bool)
for s in range(0, N_FREE, BS):
    g = m.generate(words[s:s+BS], force_valid=True)
    free_ok[s:s+BS] = (g == perms[s:s+BS]).all(1)
    print(f"[free] {s+BS}/{N_FREE} acc so far {free_ok[:s+BS].mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- stage 2: teacher-forced per-position correctness
t0 = time.time()
tf_ok = np.zeros((N_TF, 25), dtype=bool)
tf_pred = np.zeros((N_TF, 25), dtype=np.int16)
for s in range(0, N_TF, BS):
    pred, _ = m.teacher_forced_predictions(words[s:s+BS], perms[s:s+BS])
    tf_pred[s:s+BS] = pred
    tf_ok[s:s+BS] = pred == perms[s:s+BS]
    if (s // BS) % 10 == 0:
        print(f"[tf] {s+BS}/{N_TF}  pos-acc {tf_ok[:s+BS].mean():.5f}  word-acc {tf_ok[:s+BS].all(1).mean():.5f} ({time.time()-t0:.0f}s)", flush=True)

np.savez_compressed("strat_results.npz", words=words, perms=perms, hops=tr["hops"], distinct=tr["distinct"],
                    moved=moved, free_ok=free_ok, tf_ok=tf_ok, tf_pred=tf_pred)
print("DONE", flush=True)
