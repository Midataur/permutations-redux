"""
Test 14: ablate groups of heads together (torn-bigger-13, word positions), to see whether the hop-linking heads of section 3.12 back each other up.
Single heads are in head_ablation_general_results.json; here, on the SAME planted words (same seed and order as run_head_ablation.py, 200 words
per L), every non-empty subset of
   next-hop heads     (block, head): b1h3, b2h5, b2h0, b3h4
   previous-hop heads                b1h0, b2h3, b3h5
is zeroed at the word positions together, plus all seven, plus a control of four heads whose single ablation costs nothing.
If the heads were interchangeable copies, a pair would cost no more than a single head; if each does its own part, the cost adds up
(compare with the product of the single accuracies, which is what independent failures would give).
Usage: python run_group_ablation.py [N_WORDS]      -> group_ablation_general_results.json
"""
import itertools, json, sys, time
import numpy as np
from attn_tools import *
from np_transformer import layer_norm, softmax

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 200
m, cfg = load("general")
N, H, NL = cfg["N"], m.n_head, m.n_blocks
NEXT = [(0, 3), (1, 5), (1, 0), (2, 4)]
PREV = [(0, 0), (1, 3), (2, 5)]
CONTROL = [(0, 1), (1, 1), (2, 2), (1, 4)]


def forward_multi(m, toks, heads, N):
    """m.forward with the outputs of the given (layer, head) pairs set to zero at the word positions (0..N-1)."""
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


def subsets(g):
    return [list(c) for r in range(1, len(g) + 1) for c in itertools.combinations(g, r)]


groups = [("next", c) for c in subsets(NEXT)] + [("prev", c) for c in subsets(PREV)[len(PREV):]] + [("prev", [h]) for h in PREV] \
       + [("all seven", NEXT + PREV), ("control (4 free heads)", CONTROL)]
single = json.load(open("head_ablation_general_results.json"))
rng = np.random.default_rng(41)                                  # same seed and L order as run_head_ablation.py -> the same words
t0 = time.time()
results = dict(n_words=NW, groups=[], by_L={})
for L in cfg["Ls"]:
    words, ks, perm, p_ok = planted_words("general", m, rng, NW, L)
    B = len(words); toks = make_toks(m, words, perm); tgt = perm[np.arange(B), ks]
    sa = np.array(single["by_L"][str(L)]["acc_word"])
    res = {}
    for name, hs in groups:
        pred = np.zeros(B, int)
        for s in range(0, B, 50):
            lg = forward_multi(m, toks[s:s + 50], hs, N)
            pred[s:s + 50] = lg[np.arange(len(lg)), ks[s:s + 50]].argmax(-1) - m.num_trans
        key = name + ": " + " ".join(f"b{l + 1}h{h}" for l, h in hs)
        acc = float((pred == tgt).mean())
        indep = float(np.prod([sa[l, h] for l, h in hs]))
        res[key] = dict(acc=acc, independent=indep, min_single=float(min(sa[l, h] for l, h in hs)), n=len(hs))
    results["by_L"][str(L)] = dict(n_words=B, model_right_on_planted=p_ok, groups=res)
    print(f"[{time.time()-t0:4.0f}s] L={L}: {B} words", flush=True)
    for key, r in res.items():
        print(f"   {key:46s} accuracy {r['acc']:.2f}   (best single of the group {r['min_single']:.2f}; independent failures would give {r['independent']:.2f})", flush=True)
    json.dump(results, open("group_ablation_general_results.json", "w"), indent=1)
print("DONE")
