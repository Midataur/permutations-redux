"""
Planted-chain accuracy of one checkpoint (teacher-forced, at the planted slot), the same cells as sections 3.3 and 3.8 plus longer ones.
  general:    (L hops, s = min(L+1, 10) points) for L = 4..24 (inside the 10-point support), and fixed L with s beyond the support
              (s = 11, 12, 17, 21, 25).  A word has 25 letters, so L = 24 is the longest chase there is.
  elementary: width 8 and L = 12..118 (the longest training chase has 78 hops), and fixed L with other widths (training widths are <= 11).
Usage: python run_arm_accuracy.py MODEL N_PER_CELL      (MODEL = general | elem; see ckpt_env.py for other checkpoints)
       -> arm_accuracy_<MODEL><tag>_results.json
"""
import json, sys, time
import numpy as np
from attn_tools import load
from ckpt_env import SUF

kind = sys.argv[1]
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 300
m, cfg = load(kind)
rng = np.random.default_rng(2929)
t0 = time.time()
res = []
if kind == "general":
    from permdata import plant_batch
    from evalutil import tf_eval
    cells = [("in support", L, min(L + 1, 10)) for L in (4, 8, 12, 14, 16, 18, 20, 22, 24)] \
          + [("beyond support", 16, s) for s in (11, 12, 17)] + [("beyond support", 20, s) for s in (11, 12, 17, 21)] \
          + [("beyond support", 24, s) for s in (12, 17, 25)]
    for name, L, s in cells:
        w, k = plant_batch(rng, NC, L, s)
        _, correct, margin, _ = tf_eval(m, w)
        ok = int(correct[np.arange(NC), k].sum())
        res.append(dict(series=name, L=L, s=s, n=NC, ok=ok, margin_median=float(np.median(margin[np.arange(NC), k]))))
        print(f"[{time.time()-t0:4.0f}s] {name:15s} L={L:2d} s={s:2d}: {ok}/{NC} ({100*ok/NC:.1f}%)", flush=True)
else:
    from elem_tools import plant_elem_batch, tf_eval_elem
    cells = [("width 8", L, 8) for L in (12, 24, 40, 56, 70, 78, 80, 90, 100, 110, 118)] \
          + [("L=24", 24, w) for w in (4, 8, 11, 12, 14, 16)] + [("L=40", 40, w) for w in (8, 11, 12, 16)]
    for name, L, w_ in cells:
        words, ks = plant_elem_batch(rng, NC, L, w_)
        _, correct, margin = tf_eval_elem(m, words)
        ok = int(correct[np.arange(NC), ks].sum())
        res.append(dict(series=name, L=L, w=w_, n=NC, ok=ok, margin_median=float(np.median(margin[np.arange(NC), ks]))))
        print(f"[{time.time()-t0:4.0f}s] {name:9s} L={L:3d} w={w_:2d}: {ok}/{NC} ({100*ok/NC:.1f}%)", flush=True)
json.dump(dict(model=cfg["model"], blocks=m.n_blocks, cells=res), open(f"arm_accuracy_{kind}{SUF}_results.json", "w"), indent=1)
print("DONE")
