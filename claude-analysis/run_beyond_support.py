"""
Test B2: does having MORE distinct points than the training support (m = ACTUAL_GROUP_SIZE = 16 in the
checkpoint config) hurt, once the number of hops L is held fixed?

The ladder cells in run_ladder.py confound s and L (a chase with s distinct points needs L >= s-1 hops).
Here L is fixed at 16 and at 20 and s is varied across the support boundary (s <= 16 inside, s >= 17 outside).
Teacher-forced accuracy at the planted slot only (1 CPU, so no free-running decoding here).

Usage: python run_beyond_support.py N_TF      (I used 1000; about 3 minutes)
"""
from paths import MODEL_TORN
import csv, sys, time
import numpy as np
from np_transformer import NPTransformer
from permdata import plant_batch
from evalutil import tf_eval

NTF = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
m = NPTransformer(MODEL_TORN)
rng = np.random.default_rng(2027)

cells = [(16, s) for s in (6, 12, 16, 17)] + [(20, s) for s in (6, 12, 16, 17, 21)]

t0 = time.time()
with open("beyond_support_results.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["L_hops", "s_distinct", "n_tf", "tf_k_ok", "tf_margin_median"])
    for L, s in cells:
        w, k = plant_batch(rng, NTF, L, s)
        _, correct, margin, _ = tf_eval(m, w)
        idx = np.arange(NTF)
        ok = int(correct[idx, k].sum())
        wr.writerow([L, s, NTF, ok, round(float(np.median(margin[idx, k])), 2)])
        f.flush()
        print(f"[{time.time()-t0:4.0f}s] L={L:2d} s={s:2d}  TF p_k {ok}/{NTF} ({100*ok/NTF:.1f}%)  "
              f"margin med {np.median(margin[idx, k]):.1f}", flush=True)
print("DONE")
