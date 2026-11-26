"""
Test 15: how many heads must be removed together before a checkpoint fails?  (word-position heads, planted chains of the longest L tested)
Greedy search: start with no head removed; at each step zero the one additional head (output at the word positions) that lowers the
accuracy at the planted slot the most; stop after MAXK heads or when accuracy is below 15%.  Single-head ablation (section 3.12) says how
much each head matters alone; this says how much redundancy there is: a model that survives losing any one head can still collapse when
a small set is removed.
Usage: python run_greedy_ablation.py MODEL N_WORDS MAXK    (MODEL = general | elem; checkpoint through ckpt_env.py)
       -> greedy_ablation_<MODEL><tag>_results.json
"""
import json, sys, time
import numpy as np
from attn_tools import *
from ckpt_env import SUF
from np_transformer import layer_norm, softmax

kind = sys.argv[1]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 150
MAXK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
m, cfg = load(kind)
N, H, NL = cfg["N"], m.n_head, m.n_blocks


def forward_multi(m, toks, heads, N):
    B, T = toks.shape
    x = m.E[toks] + m.P[:T][None]
    M = m.mask(T)
    for li, ly in enumerate(m.layers):
        a = layer_norm(x, ly["ln1w"], ly["ln1b"])
        q = (a @ ly["Wq"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        k = (a @ ly["Wk"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        v = (a @ ly["Wv"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        s = np.where(M[None, None] == 0, -np.inf, (q @ k.transpose(0, 1, 3, 2)) * m.scale)
        o = (softmax(s) @ v).transpose(0, 2, 1, 3)
        for (l2, h2) in heads:
            if l2 == li:
                o[:, :N, h2, :] = 0.0
        x = x + (o.reshape(B, T, m.D) @ ly["Wo"].T + ly["bo"])
        f = layer_norm(x, ly["ln2w"], ly["ln2b"])
        x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
    return layer_norm(x, m.lnfw, m.lnfb)[:, -m.max_group:, :] @ m.Wlm.T


L = cfg["Ls"][-1]
rng = np.random.default_rng(77)
words, ks, perm, p_ok = planted_words(kind, m, rng, NW, L)
B = len(words); toks = make_toks(m, words, perm); tgt = perm[np.arange(B), ks]


def acc_of(heads):
    pred = np.zeros(B, int)
    for s in range(0, B, 50):
        lg = forward_multi(m, toks[s:s + 50], heads, N)
        pred[s:s + 50] = lg[np.arange(len(lg)), ks[s:s + 50]].argmax(-1) - m.num_trans
    return float((pred == tgt).mean())


cands = [(l, h) for l in range(NL - 1) for h in range(H)]            # the last block is never read at the word positions
chosen, steps = [], []
t0 = time.time()
print(f"{kind}{SUF}: L={L}, {B} words kept ({100*p_ok:.0f}% of planted right), {len(cands)} candidate heads", flush=True)
for k in range(MAXK):
    best = min(((acc_of(chosen + [c]), c) for c in cands if c not in chosen), key=lambda x: x[0])
    chosen.append(best[1]); steps.append(dict(k=k + 1, added=f"b{best[1][0] + 1}h{best[1][1]}", acc=best[0]))
    print(f"[{time.time()-t0:4.0f}s] remove {k + 1}: + b{best[1][0] + 1}h{best[1][1]}  ->  accuracy {best[0]:.2f}", flush=True)
    if best[0] < 0.15:
        break
json.dump(dict(model=cfg["model"], L=L, n_words=B, steps=steps), open(f"greedy_ablation_{kind}{SUF}_results.json", "w"), indent=1)
print("DONE")
