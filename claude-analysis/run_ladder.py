"""
Test B1: planted "ladder" trajectories. For one output position k we force the pointer chase to make exactly
L hops and visit exactly s distinct points (incl. start); all non-hop letters are random but avoid the
pointer's current position. Other positions of the word keep whatever (typically easy) trajectories result.

Per cell we report
  free_word : exact-match of the full permutation under free-running greedy decoding (repo metric)
  free_k    : correctness of the planted entry p_k in the free-running output
  tf_k      : correctness of p_k under teacher forcing, plus logit margin (true minus best competitor)
Layouts: 'spread' = hop letters at random word positions; 'contig' = hops occupy the last L letters of the word.
"""
from paths import MODEL_TORN
import csv, time, sys
import numpy as np
from np_transformer import NPTransformer
from permdata import *
from evalutil import tf_eval

MODEL = MODEL_TORN
NFREE, NTF = int(sys.argv[1]), int(sys.argv[2])
m = NPTransformer(MODEL)
rng = np.random.default_rng(2026)
n = N = 25

cells = []
for s in [4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 20, 24]:
    cells.append(("ladder_spread", "spread", s - 1, s))        # minimal hops: every hop reaches a new point
for s in [4, 6, 8, 10]:
    cells.append(("control_manyhops", "spread", 20, s))        # many hops, few distinct points
for L in [14, 17, 20]:
    cells.append(("hops_at_s12", "spread", L, 12))             # same support, more hops
for s in [8, 10, 11, 12, 16]:
    cells.append(("ladder_contig", "contig", s - 1, s))

contig = lambda rng, L, N: np.arange(L)
rows = []
t00 = time.time()
with open("ladder_results.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["cell", "layout", "L_hops", "s_distinct", "n_free", "free_word_ok", "free_k_ok",
                 "n_tf", "tf_k_ok", "tf_margin_median", "tf_margin_min", "tf_other_positions_ok_frac"])
    for name, layout, L, s in cells:
        fn = contig if layout == "contig" else None
        wf, kf = plant_batch(rng, NFREE, L, s, hop_reads_fn=fn)
        pf = apply_words(wf)
        fw = np.zeros(NFREE, bool); fk = np.zeros(NFREE, bool)
        for a in range(0, NFREE, 100):
            g = m.generate(wf[a:a+100], force_valid=True)
            fw[a:a+100] = (g == pf[a:a+100]).all(1)
            fk[a:a+100] = g[np.arange(len(g)), kf[a:a+100]] == pf[a:a+100][np.arange(len(g)), kf[a:a+100]]
        wt, kt = plant_batch(rng, NTF, L, s, hop_reads_fn=fn)
        perms, correct, margin, pred = tf_eval(m, wt)
        idx = np.arange(NTF)
        tk = correct[idx, kt]
        oth = np.ones_like(correct); oth[idx, kt] = False
        r = [name, layout, L, s, NFREE, int(fw.sum()), int(fk.sum()), NTF, int(tk.sum()),
             round(float(np.median(margin[idx, kt])), 2), round(float(margin[idx, kt].min()), 2),
             round(float(correct[oth].mean()), 5)]
        wr.writerow(r); f.flush(); rows.append(r)
        print(f"[{time.time()-t00:5.0f}s] {name:17s} L={L:2d} s={s:2d}  free word {fw.sum()}/{NFREE}  free p_k {fk.sum()}/{NFREE}  "
              f"TF p_k {tk.sum()}/{NTF}  margin med {np.median(margin[idx,kt]):.1f} min {margin[idx,kt].min():.1f}", flush=True)
print("DONE", flush=True)
