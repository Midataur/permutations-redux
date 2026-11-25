"""
Test 6b: position-embedding ablation on the elementary model (the model of Figure 5b; blocks of word positions at [0,72), [72,120)).
The embedding table is edited in memory: the letter at word position i is given the position vector of sigma(i).
If the model reads order from the embeddings, its answers should follow the REARRANGED word w' (w'[sigma(i)] = w[i]).
Per slot where perm(w) != perm(w'): fraction of predictions equal to perm(w) ("original"), to perm(w') ("rearranged"), or neither.
Teacher forcing with the original word's true outputs. Real test words (test_data.csv).
Usage: python run_pos_ablation.py DATA_DIR N_WORDS_SEG N_WORDS_PAIR
"""
import sys, json, time
import numpy as np, pandas as pd
from np_transformer import NPTransformer
from elem_tools import *

D, NS, NP_ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
MODEL = "../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
m = NPTransformer(MODEL, max_group=G, input_len=N, num_trans=G)
P0 = m.P.copy()
W = pd.read_csv(f"{D}/test_data.csv", header=None).to_numpy(dtype=np.int64)
t0 = time.time()


def run(words, sigma):
    """sigma: array of length 120 (a permutation of positions)."""
    perms = apply_words_elem(words)
    w2 = np.empty_like(words); w2[:, sigma] = words                 # w'[sigma(i)] = w[i]
    perms2 = apply_words_elem(w2)
    P = P0.copy(); P[:N] = P0[sigma]                                # letter at i gets the vector of sigma(i)
    m.P = P
    pred = np.concatenate([m.teacher_forced_predictions(words[s:s + 50], perms[s:s + 50])[0] for s in range(0, len(words), 50)])
    m.P = P0
    diff = perms != perms2
    n = int(diff.sum())
    return dict(n_slots=n, orig=float((pred == perms)[diff].mean()) if n else None, rearr=float((pred == perms2)[diff].mean()) if n else None,
                neither=float(((pred != perms) & (pred != perms2))[diff].mean()) if n else None,
                acc_all_slots=float((pred == perms).mean()))


def seg_swap(a, b, L):
    s = np.arange(N); s[a:a + L] = np.arange(b, b + L); s[b:b + L] = np.arange(a, a + L); return s


res = {}
base = run(W[:NS], np.arange(N)); res["identity"] = base
print(f"identity: accuracy {base['acc_all_slots']:.4f}", flush=True)
segs = {"within block 1, adjacent  [0,24)<->[24,48)": (0, 24), "within block 1, far  [0,24)<->[48,72)": (0, 48),
        "within block 2, adjacent  [72,96)<->[96,120)": (72, 96), "across 72, adjacent  [48,72)<->[72,96)": (48, 72),
        "across 72, far  [24,48)<->[96,120)": (24, 96), "across 72, far  [0,24)<->[96,120)": (0, 96)}
for name, (a, b) in segs.items():
    r = run(W[:NS], seg_swap(a, b, 24)); res["seg: " + name] = r
    print(f"[{time.time()-t0:4.0f}s] {name}: slots {r['n_slots']}  original {r['orig']:.3f}  rearranged {r['rearr']:.3f}  neither {r['neither']:.3f}", flush=True)
# sweep of single adjacent-position swaps (i, i+1)
sw = []
for i in range(N - 1):
    s = np.arange(N); s[i], s[i + 1] = i + 1, i
    r = run(W[:NP_], s); r["i"] = i; sw.append(r)
    if i % 10 == 0: print(f"[{time.time()-t0:4.0f}s] adjacent swap ({i},{i+1}): original {r['orig']}, rearranged {r['rearr']}, n {r['n_slots']}", flush=True)
res["adjacent_swaps"] = sw
json.dump(res, open("pos_ablation_results.json", "w"), indent=1)
print("DONE", flush=True)
