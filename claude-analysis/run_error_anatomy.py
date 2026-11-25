"""When the model fails on a long planted chase, what does it output? Compare the teacher-forced prediction at the
planted slot with the pointer's position after r hops (r = 0..L)."""
from paths import MODEL_TORN
import numpy as np, csv
from np_transformer import NPTransformer
from permdata import *
from evalutil import tf_eval

m = NPTransformer(MODEL_TORN)
rng = np.random.default_rng(7)
out = []
for name, L, s in [("ladder", 19, 20), ("ladder", 23, 24), ("many hops, few points", 20, 4), ("many hops, s=12", 20, 12), ("moderate", 17, 12)]:
    w, k = plant_batch(rng, 600, L, s)
    perms, correct, margin, pred = tf_eval(m, w)
    idx = np.arange(len(w))
    fail = ~correct[idx, k]
    short = []          # hops short of the end: L - r for the (earliest-ending) r with path[r]==pred
    off_path = 0; nonperm = 0
    for i in np.nonzero(fail)[0]:
        p = int(pred[i, k[i]])
        if not (0 <= p < 25):
            nonperm += 1; continue
        path = pointer_path(w[i], k[i])
        rs = [r for r, x in enumerate(path) if x == p]
        if not rs:
            off_path += 1
        else:
            short.append(L - max(rs))      # hops short of the true end (smallest shortfall consistent with pred)
    short = np.array(short)
    print(f"\n{name}: L={L} s={s}: fail {fail.sum()}/{len(w)}  | pred off the chase path: {off_path}  non-permutation token: {nonperm}  on path: {len(short)}")
    if len(short):
        vals, cnt = np.unique(short, return_counts=True)
        print("   hops short of the true end (L - r) -> count:", dict(zip(vals.tolist(), cnt.tolist())))
        print(f"   median shortfall {np.median(short):.0f};  share short by exactly 1 hop {np.mean(short==1):.2f};  short by <=2: {np.mean(short<=2):.2f}")
