"""
Test 6a, part 2: elementary model, planted (hops L, width w) cells and natural stratified accuracy.
w = number of distinct positions the pointer visits (an interval). Training windows give w <= 11 (real data: elem_stats.txt);
the real test file has widths up to 15. Teacher-forced accuracy at the planted slot.
Usage: python run_elem_cells.py N_PER_CELL N_NATURAL
"""
import sys, csv, time
import numpy as np
from np_transformer import NPTransformer
from elem_tools import *

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 500
NNAT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
MODEL = "../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
m = NPTransformer(MODEL, max_group=G, input_len=N, num_trans=G)
rng = np.random.default_rng(4242)
t0 = time.time()

cells = [("A fixed w=8", L, 8) for L in (12, 24, 40, 56, 80)] \
      + [("B fixed L=24", 24, w) for w in (4, 8, 11, 12, 14, 16)] \
      + [("C fixed L=40", 40, w) for w in (8, 11, 12, 16)]
with open("elem_cells_results.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["series", "L_hops", "w_width", "n_tf", "tf_k_ok", "tf_margin_median"])
    for name, L, w in cells:
        words, ks = plant_elem_batch(rng, NC, L, w)
        _, correct, margin = tf_eval_elem(m, words)
        idx = np.arange(NC)
        ok = int(correct[idx, ks].sum())
        wr.writerow([name, L, w, NC, ok, round(float(np.median(margin[idx, ks])), 2)]); f.flush()
        print(f"[{time.time()-t0:4.0f}s] {name:13s} L={L:2d} w={w:2d}: TF {ok}/{NC} ({100*ok/NC:.1f}%)  margin med {np.median(margin[idx, ks]):.1f}", flush=True)

# natural words: 120 uniform letters over tokens 0..15 (same as the real test file)
W = rng.integers(0, G, size=(NNAT, N))
tr = trajectories_elem(W, check=True)
_, correct, margin = tf_eval_elem(m, W)
print(f"\n[{time.time()-t0:4.0f}s] natural words: {NNAT} words, {correct.size} slots; slot acc {correct.mean():.5f}, word acc {correct.all(1).mean():.4f}")
hops, width = tr["hops"].ravel(), tr["width"].ravel(); c = correct.ravel()
print("accuracy by width (slots, errors):")
for w in range(1, G + 1):
    s = width == w
    if s.any(): print(f"   w={w:2d}: {s.sum():8d} slots, {int((~c[s]).sum()):5d} errors  ({100*c[s].mean():.3f}% correct)")
print("accuracy by hop bin (slots, errors):")
for a, b in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 121)):
    s = (hops >= a) & (hops < b)
    if s.any(): print(f"   hops {a:3d}-{b-1:3d}: {s.sum():8d} slots, {int((~c[s]).sum()):5d} errors  ({100*c[s].mean():.3f}% correct)")
print("accuracy for wide pointers (w >= 12) by hop bin:")
for a, b in ((0, 15), (15, 25), (25, 121)):
    s = (width >= 12) & (hops >= a) & (hops < b)
    if s.any(): print(f"   w>=12, hops {a:3d}-{b-1:3d}: {s.sum():6d} slots, {int((~c[s]).sum()):4d} errors ({100*c[s].mean():.2f}% correct)")
np.savez("elem_natural.npz", hops=hops, width=width, correct=c)
print("DONE")
