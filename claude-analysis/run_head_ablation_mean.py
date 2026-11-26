"""
Test 12b: the same single-head ablation as run_head_ablation.py, but the head's output is REPLACED BY ITS MEAN over other planted
words (separate sample, same L, per position) instead of zero.  This keeps what the head adds that does not depend on the word
(its position-dependent constant) and removes only what depends on this word, so a drop here is closer to "the head's
word-specific content is needed".  Run only for the heads and variants whose zero-ablation accuracy at that L is below 0.92
(read from head_ablation_<MODEL>_results.json).

Usage: python run_head_ablation_mean.py MODEL N_WORDS N_MEAN_WORDS     -> head_ablation_mean_<MODEL>_results.json
"""
import json, sys, time
import numpy as np
from attn_tools import *

kind = sys.argv[1]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NM = int(sys.argv[3]) if len(sys.argv) > 3 else 150
m, cfg = load(kind)
N, H, NL = cfg["N"], m.n_head, m.n_blocks
Z = json.load(open(f"head_ablation_{kind}_results.json"))
rng = np.random.default_rng({"general": 51, "elem": 52}[kind])
t0 = time.time()
results = dict(model=cfg["model"], N=N, threshold=0.92, by_L={})
for L in cfg["Ls"]:
    # the words to test are the same ones run_head_ablation.py used (same seed, same draw order is not guaranteed here: fresh draw)
    mw, mk, mp, _ = planted_words(kind, m, rng, NM, L)
    mt = make_toks(m, mw, mp); T = mt.shape[1]
    sums = None
    for s in range(0, len(mt), 50):
        _, outs = forward_ablate(m, mt[s:s + 50], None, collect=True)
        part = [o.sum(0) for o in outs]
        sums = part if sums is None else [a + b for a, b in zip(sums, part)]
    means = [a / len(mt) for a in sums]                                  # per layer: (T, H, hd)
    words, ks, perm, p_ok = planted_words(kind, m, rng, NW, L)
    B = len(words); toks = make_toks(m, words, perm); tgt = perm[np.arange(B), ks]
    pos = {"word": np.arange(T) < N, "output": np.arange(T) >= N}
    zero = {"word": np.array(Z["by_L"][str(L)]["acc_word"]), "output": np.array(Z["by_L"][str(L)]["acc_output"])}
    res = {"word": {}, "output": {}}
    for w, pm in pos.items():
        for l in range(NL):
            for h in range(H):
                if zero[w][l, h] >= 0.92:
                    continue
                pred = np.zeros(B, int)
                for s in range(0, B, 50):
                    lg = forward_ablate(m, toks[s:s + 50], (l, h, pm, means[l][:, h, :]))
                    pred[s:s + 50] = lg[np.arange(len(lg)), ks[s:s + 50]].argmax(-1) - m.num_trans
                res[w][f"{l},{h}"] = dict(mean=float((pred == tgt).mean()), zero=float(zero[w][l, h]))
        print(f"[{time.time()-t0:4.0f}s] {kind} L={L} {w}: " + "  ".join(f"b{k.split(',')[0] and int(k.split(',')[0])+1}h{k.split(',')[1]}: mean {v['mean']:.2f} (zero {v['zero']:.2f})" for k, v in res[w].items()), flush=True)
    results["by_L"][str(L)] = dict(n_words=B, n_mean_words=len(mt), results=res)
    json.dump(results, open(f"head_ablation_mean_{kind}_results.json", "w"))
print("DONE", flush=True)
