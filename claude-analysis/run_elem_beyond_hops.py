"""
Test 13: the elementary model beyond its longest training chase.
In the real training file (100,000 words, 1.6M slots; elem_stats.txt, elem_train_chase_hist.npz) the longest chase has 78 hops, only 8 slots
have 70 or more and 1 has 78 or more.  So the model's 99.8% at L=80 (section 3.8) is at the edge of its training chases, not far inside them.
Here: planted chases of L = 56 ... 118 hops (the word has 120 letters, so 118 means almost every letter is a hop), width w = 8 (and 4 as a
control), teacher-forced accuracy at the planted slot.  A fall past 78 would be a hop-count edge like the general model's; no fall would mean
that hop count extrapolates in this model.
Usage: python run_elem_beyond_hops.py N_PER_CELL  ->  elem_beyond_hops_results.json
"""
import sys, json, time
import numpy as np
from np_transformer import NPTransformer
from elem_tools import *

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MODEL = "../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
m = NPTransformer(MODEL, max_group=G, input_len=N, num_trans=G)
rng = np.random.default_rng(1313)
H = np.load("elem_train_chase_hist.npz")["H2"]; h = H.sum(1)
cells = [(8, L) for L in (56, 70, 78, 80, 84, 90, 100, 110, 118)] + [(4, L) for L in (80, 100, 118)]
res = []
t0 = time.time()
for w, L in cells:
    words, ks = plant_elem_batch(rng, NC, L, w)
    _, correct, margin = tf_eval_elem(m, words)
    ok = int(correct[np.arange(NC), ks].sum())
    # all-slot check on the same words: the other slots are natural-ish, so report only the planted one
    res.append(dict(w=w, L=L, n=NC, ok=ok, train_slots_with_this_many_hops=int(h[L]) if L < len(h) else 0,
                    train_slots_with_at_least=int(h[L:].sum()), margin_median=float(np.median(margin[np.arange(NC), ks]))))
    print(f"[{time.time()-t0:4.0f}s] w={w} L={L:3d}: {ok}/{NC} ({100*ok/NC:.1f}%)  margin med {res[-1]['margin_median']:.1f}   "
          f"(training slots with >= {L} hops: {res[-1]['train_slots_with_at_least']} of {int(h.sum())})", flush=True)
    json.dump(res, open("elem_beyond_hops_results.json", "w"), indent=1)
print("DONE")
