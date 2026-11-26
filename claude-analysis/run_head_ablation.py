"""
Test 12: single-head ablation on planted chains.

For every layer l and head h, zero the output of that head (a) at the WORD positions only, or (b) at the OUTPUT side only
(START and the output tokens, which include the query position that predicts the slot), and measure teacher-forced accuracy at
the planted slot on words the model gets right unablated.  (a) says whether the head is needed to build what the word positions
hold; (b) says whether it is needed to read it.  Zeroing is crude (it also removes the head's constant part), so read the drops
relative to one another rather than as an exact size of the head's job.

Usage: python run_head_ablation.py MODEL N_WORDS      (MODEL = general | elem)   -> head_ablation_<MODEL>_results.json
"""
import json, sys, time
import numpy as np
from attn_tools import *
from ckpt_env import SUF

kind = sys.argv[1]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 200
m, cfg = load(kind)
N, H, NL = cfg["N"], m.n_head, m.n_blocks
rng = np.random.default_rng({"general": 41, "elem": 42}[kind])
t0 = time.time()
results = dict(model=cfg["model"], N=N, n_heads=H, n_layers=NL, by_L={})
for L in cfg["Ls"]:
    words, ks, perm, p_ok = planted_words(kind, m, rng, NW, L)
    B = len(words); toks = make_toks(m, words, perm); T = toks.shape[1]; tgt = perm[np.arange(B), ks]
    pos = {"word": np.arange(T) < N, "output": np.arange(T) >= N}
    acc = {w: np.zeros((NL, H)) for w in pos}
    for l in range(NL):
        for h in range(H):
            for w, pm in pos.items():
                pred = np.zeros(B, int)
                for s in range(0, B, 50):
                    lg = forward_ablate(m, toks[s:s + 50], (l, h, pm))
                    pred[s:s + 50] = lg[np.arange(len(lg)), ks[s:s + 50]].argmax(-1) - m.num_trans
                acc[w][l, h] = (pred == tgt).mean()
        print(f"[{time.time()-t0:4.0f}s] {kind} L={L} layer {l+1}: word-position ablation " + " ".join(f"{a:.2f}" for a in acc['word'][l]) +
              "   output-side ablation " + " ".join(f"{a:.2f}" for a in acc['output'][l]), flush=True)
    results["by_L"][str(L)] = dict(n_words=B, model_right_on_planted=p_ok, acc_word=acc["word"].tolist(), acc_output=acc["output"].tolist())
    json.dump(results, open(f"head_ablation_{kind}{SUF}_results.json", "w"))
print("DONE", flush=True)
