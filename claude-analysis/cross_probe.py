"""
Do the general and the elementary model represent the same word the same way?  (Max's suggestion, 28 Nov 2026.)

Words: 25 letters, each an adjacent transposition s_i = (i-1, i), i = 1..15, uniform: words of S_16 in adjacent letters.
  general model (torn-bigger-13):  letter s_i is token 25*(i-1) + i; the permutation lives on points 0..15, 16..24 fixed.
  elementary model (elem-partitioned-long-boosted-1): letter s_i is token i at word positions 95..119; positions 0..94 hold
     one identity and 47 random cancelling pairs s_j s_j, which leave the permutation unchanged.  (Padding with 95
     identities instead puts the elementary model off: 88.5% of slots right, 13.7% of words; with pairs it is 100%.)
Aligned positions: the 25 letters (general 0..24 <-> elementary 95..119); START and the outputs holding p_0..p_14
(general 25..40 <-> elementary 120..135).
For every pair of layers (0 = embeddings, 1..5 = after each block): ridge regression from one model's residual stream to
the other's at aligned positions, with each position's mean removed (so position alone explains nothing), fitted on 80% of
the words and scored on the other 20%: R^2 summed over all 402 dimensions.  Both directions.
Controls: (a) the same regression with the word pairing shuffled; (b) simple features of the word: at letter positions the
letter itself, and all letters by relative offset (a linear translation-equivariant function of the word); at output positions
the tokens fed in and the answer, and the whole permutation.

Usage: python cross_probe.py N_WORDS -> cross_probe.json, cross_probe_log.txt
"""
import sys, json
import numpy as np
from np_transformer import NPTransformer
from elem_tools import apply_words_elem

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = np.random.default_rng(2026)
gen = NPTransformer("../scaling-generator/trained-models/torn-bigger-13/model.safetensors", 25, 25, n_blocks=5)
ele = NPTransformer("../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors", 16, 120, n_blocks=5, num_trans=16)

letters = rng.integers(1, 16, size=(NW, 25))                       # s_1..s_15
W_e = np.zeros((NW, 120), np.int64); W_e[:, 95:] = letters
for b in range(NW):                       # padding: 47 adjacent equal pairs s_j s_j (each cancels) and one identity
    js = rng.integers(1, 16, size=47); W_e[b, 1:95:2] = js; W_e[b, 2:95:2] = js
W_g = 25 * (letters - 1) + letters
P16 = apply_words_elem(W_e)                                         # (NW,16)
P_g = np.concatenate([P16, np.tile(np.arange(16, 25), (NW, 1))], 1)
# check the general model's convention gives the same permutation
from permdata import apply_words
assert np.array_equal(apply_words(W_g, 25), P_g)


def run(model, W, P, keep, bs=100):
    hid, preds = [], []
    for s in range(0, len(W), bs):
        seq, perm = W[s:s + bs], P[s:s + bs]
        start = np.full((len(seq), 1), model.START)
        toks = np.concatenate([seq, start, perm[:, :-1] + model.num_trans], 1)
        logits, hidden = model.forward(toks, return_hidden=True)
        preds.append(logits.argmax(-1) - model.num_trans)
        hid.append(np.stack([h[:, keep] for h in hidden], 0).astype(np.float32))   # (layers+1, B, kept positions, D)
    return np.concatenate(preds), np.concatenate(hid, 1)


KG = np.arange(0, 41); KE = np.concatenate([95 + np.arange(25), 120 + np.arange(16)])
pg, Hg = run(gen, W_g, P_g, KG); pe, He = run(ele, W_e, P16, KE)
acc_g = float((pg[:, :16] == P16).mean()); acc_e = float((pe == P16).mean())
acc_gw = float((pg[:, :16] == P16).all(1).mean()); acc_ew = float((pe == P16).all(1).mean())
print(f"accuracy on these words: general {acc_g:.4f} per slot ({acc_gw:.3f} per word, first 16 slots); elementary {acc_e:.4f} ({acc_ew:.3f})", flush=True)

groups = {"letters": (np.arange(25), np.arange(25)), "outputs": (25 + np.arange(16), 25 + np.arange(16))}   # indices into the kept positions
ntr = int(0.8 * NW)


def centred(X, pos):
    """X: (B, T, D) -> (B*len(pos), D) with each position's training-set mean removed"""
    Y = X[:, pos]
    return (Y - Y[:ntr].mean(0, keepdims=True))


def ridge_r2(Xtr, Ytr, Xte, Yte, lams=(1e-2, 1e-1, 1, 10, 100)):
    """fit on half of the training words for the penalty, refit on all, score R^2 on the test words"""
    best = None
    k = len(Xtr) // 2
    for lam in lams:
        A = Xtr[:k]; Wt = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) * 1e-3, A.T @ Ytr[:k])
        r = 1 - ((Xtr[k:] @ Wt - Ytr[k:]) ** 2).sum() / ((Ytr[k:] - Ytr[k:].mean(0)) ** 2).sum()
        if best is None or r > best[0]:
            best = (r, lam)
    lam = best[1]
    Wt = np.linalg.solve(Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]) * 1e-3, Xtr.T @ Ytr)
    return float(1 - ((Xte @ Wt - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())


def flat(Z):           # (B, P, D) -> (B*P, D)
    return Z.reshape(-1, Z.shape[-1])


L = Hg.shape[0]
out = dict(n_words=NW, accuracy=dict(general_slot=acc_g, general_word=acc_gw, elementary_slot=acc_e, elementary_word=acc_ew), groups={})
perm_words = rng.permutation(NW)
oh = np.eye(16)[letters][:, :, 1:]                                      # (NW, 25, 15)
# letters: (i) the letter at the position; (ii) letters by relative offset -24..+24 (zeros outside the word): a linear,
# translation-equivariant function of the word.  outputs: (i) the token fed in at the position (p_{k-1}) and the answer p_k;
# (ii) the whole permutation (one-hot of all 16 entries).
rel = np.zeros((NW, 25, 49, 15))
for t_ in range(25):
    for o in range(-24, 25):
        if 0 <= t_ + o < 25:
            rel[:, t_, o + 24] = oh[:, t_ + o]
P1 = np.eye(16)[P16]                                                     # (NW, 16, 16)
fed = np.concatenate([np.zeros((NW, 1, 16)), P1[:, :15]], 1)             # token fed in at output position k: START, p_0..p_14
CTRL = {"letters": {"own_letter": oh, "letters_by_offset": rel.reshape(NW, 25, -1)},
        "outputs": {"fed_and_answer": np.concatenate([fed, P1], 2), "whole_permutation": np.repeat(P1.reshape(NW, 1, -1), 16, 1)}}
for gname, (pos_g, pos_e) in groups.items():
    G = [centred(Hg[l], pos_g) for l in range(L)]; E_ = [centred(He[l], pos_e) for l in range(L)]
    npos = len(pos_g)
    res = {"g_to_e": np.zeros((L, L)), "e_to_g": np.zeros((L, L)), "g_to_e_shuffled": np.zeros(L), "e_to_g_shuffled": np.zeros(L),
           }
    for a in range(L):
        for b in range(L):
            Xa, Yb = G[a], E_[b]
            res["g_to_e"][a, b] = ridge_r2(flat(Xa[:ntr]), flat(Yb[:ntr]), flat(Xa[ntr:]), flat(Yb[ntr:]))
            Xa, Yb = E_[a], G[b]
            res["e_to_g"][a, b] = ridge_r2(flat(Xa[:ntr]), flat(Yb[:ntr]), flat(Xa[ntr:]), flat(Yb[ntr:]))
        # shuffled pairing at the same layer
        Gs = G[a][perm_words]
        res["g_to_e_shuffled"][a] = ridge_r2(flat(Gs[:ntr]), flat(E_[a][:ntr]), flat(Gs[ntr:]), flat(E_[a][ntr:]))
        Es = E_[a][perm_words]
        res["e_to_g_shuffled"][a] = ridge_r2(flat(Es[:ntr]), flat(G[a][:ntr]), flat(Es[ntr:]), flat(G[a][ntr:]))
        # controls: what simple features of the word explain at these positions
        for cname, X1 in CTRL[gname].items():
            res.setdefault(f"{cname}_to_g", np.zeros(L)); res.setdefault(f"{cname}_to_e", np.zeros(L))
            X1 = X1 - X1[:ntr].mean(0, keepdims=True)
            res[f"{cname}_to_g"][a] = ridge_r2(flat(X1[:ntr]), flat(G[a][:ntr]), flat(X1[ntr:]), flat(G[a][ntr:]))
            res[f"{cname}_to_e"][a] = ridge_r2(flat(X1[:ntr]), flat(E_[a][:ntr]), flat(X1[ntr:]), flat(E_[a][ntr:]))
        print(gname, "layer", a, "done", flush=True)
    out["groups"][gname] = {k: v.tolist() for k, v in res.items()}
    print(f"\n{gname}: R^2 predicting the elementary model (columns: its layer 0..5) from the general model (rows: layer 0..5)")
    print(np.array2string(res["g_to_e"], precision=2, suppress_small=True))
    print("reverse direction (rows: elementary layer, columns: general layer)")
    print(np.array2string(res["e_to_g"], precision=2, suppress_small=True))
    print("same layer, shuffled pairing g->e:", np.round(res["g_to_e_shuffled"], 3), " e->g:", np.round(res["e_to_g_shuffled"], 3))
    for cname in CTRL[gname]:
        print(f"from {cname}: general", np.round(res[f"{cname}_to_g"], 3), " elementary", np.round(res[f"{cname}_to_e"], 3), flush=True)
json.dump(out, open("cross_probe.json", "w"), indent=1)
