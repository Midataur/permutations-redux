"""
Test 11: attention patterns at the word positions and at the query position, for planted chains.

For every layer and head:
 (1) WORD POSITIONS.  For each hop letter j of slot k's chain (j = 1 is the one read first), the share of its attention
     (a word position attends to word positions only) that falls on each kind of letter:
         self | next hop letter (j+1, read later, to the left) | previous hop letter (j-1, read earlier, to the right)
         | other hop letters | non-chain letters that share a point with this letter | all other letters.
     The "expected" share is what uniform attention over the 25 (or 120) word positions would give; ratio = share / expected.
 (2) QUERY POSITION (the position that predicts slot k).  Its attention to hop letter j, for every j, to the non-chain
     word letters, and to START and the earlier output tokens.

Usage: python run_attention.py MODEL N_WORDS      (MODEL = general | elem)   -> attention_<MODEL>_results.json
"""
import json, sys, time
import numpy as np
from attn_tools import *
from ckpt_env import SUF

kind = sys.argv[1]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 200
m, cfg = load(kind)
N, H, NL = cfg["N"], m.n_head, m.n_blocks
rng = np.random.default_rng({"general": 31, "elem": 32}[kind])
CATS = ["self", "next", "prev", "chain_other", "touch", "rest"]
t0 = time.time()
results = dict(model=cfg["model"], N=N, n_heads=H, n_layers=NL, categories=CATS, by_L={})

for L in cfg["Ls"]:
    words, ks, perm, p_ok = planted_words(kind, m, rng, NW, L)
    B = len(words)
    hops = [hop_letters(kind, w, k) for w, k in zip(words, ks)]
    assert all(len(h) == L for h in hops)
    EA, EB = endpoints(kind, words)
    toks = make_toks(m, words, perm)
    T = toks.shape[1]
    word_mass = np.zeros((NL, H, len(CATS))); word_exp = np.zeros(len(CATS))
    q_hop = np.zeros((NL, H, L)); q_non = np.zeros((NL, H)); q_rest = np.zeros((NL, H)); q_exp_chain = 0.0
    OFFS = list(range(-8, 17))                                       # hop offset d: letter j attends to hop letter j+d
    off_sum = np.zeros((NL, H, len(OFFS))); off_cnt = np.zeros(len(OFFS))
    pos_attn = np.zeros((NL, H, N)); pos_chain_sum = np.zeros((NL, H, N)); pos_non_sum = np.zeros((NL, H, N))
    chain_cnt = np.zeros(N); non_cnt = np.zeros(N)
    for s in range(0, B, 50):
        _, attns = m.forward(toks[s:s + 50], return_attn=True)
        for bi in range(len(toks[s:s + 50])):
            i = s + bi
            ts = np.array([h[0] for h in hops[i]])                   # positions of hop letters, reading order j = 1..L
            lab = np.full((L, N), CATS.index("rest"), int)
            share = ((EA[i][None] == EA[i][ts][:, None]) | (EA[i][None] == EB[i][ts][:, None]) |
                     (EB[i][None] == EA[i][ts][:, None]) | (EB[i][None] == EB[i][ts][:, None])) & (EA[i][None] >= 0)
            lab[share] = CATS.index("touch")
            lab[:, ts] = CATS.index("chain_other")
            for j in range(L):
                lab[j, ts[j]] = CATS.index("self")
                if j + 1 < L: lab[j, ts[j + 1]] = CATS.index("next")
                if j > 0: lab[j, ts[j - 1]] = CATS.index("prev")
            onehot = np.stack([(lab == c) for c in range(len(CATS))])   # (C, L, N)
            # expected share under uniform attention, averaged over the rows that have a letter of the kind (as word_mass is)
            word_exp += np.array([onehot[c].sum(-1)[onehot[c].any(-1)].mean() for c in range(len(CATS))]) / N / B
            ischain = np.zeros(N, bool); ischain[ts] = True
            chain_cnt += ischain; non_cnt += ~ischain
            for di, d in enumerate(OFFS):
                if abs(d) < L: off_cnt[di] += 1
            for l in range(NL):
                P = attns[l][bi]                                      # (H, T, T)
                R = P[:, ts, :N]                                      # (H, L, N)
                A = R[:, :, ts]                                       # (H, L, L): hop j's attention to hop j'
                for di, d in enumerate(OFFS):
                    if abs(d) < L:
                        off_sum[l, :, di] += A.diagonal(offset=d, axis1=1, axis2=2).mean(-1)
                qn = P[:, N + ks[i], :N]
                pos_attn[l] += qn; pos_chain_sum[l] += qn * ischain; pos_non_sum[l] += qn * ~ischain
                assert np.allclose(P[:, ts, N:].sum(), 0.0)           # word rows never look at the outputs
                for c in range(len(CATS)):
                    msk = onehot[c]
                    per_row = (R * msk[None]).sum(-1)                 # (H, L)
                    has = msk.any(-1)                                 # rows that have a letter of this kind
                    word_mass[l, :, c] += per_row[:, has].mean(-1) / B
                q = P[:, N + ks[i], :]                                # (H, T)
                q_hop[l] += q[:, ts] / B
                nonchain = np.ones(N, bool); nonchain[ts] = False
                q_non[l] += q[:, :N][:, nonchain].sum(-1) / B
                q_rest[l] += q[:, N:].sum(-1) / B
    q_exp_chain = L / N
    results["by_L"][str(L)] = dict(n_words=B, model_right_on_planted=p_ok, word_mass=word_mass.tolist(), word_expected=word_exp.tolist(),
                                   query_hop=q_hop.tolist(), query_nonchain=q_non.tolist(), query_other=q_rest.tolist(),
                                   query_expected_chain=q_exp_chain, offsets=OFFS,
                                   offset_mass=(off_sum / np.maximum(off_cnt, 1)).tolist(),
                                   chain_freq=(chain_cnt / B).tolist(), query_pos_attn=(pos_attn / B).tolist(),
                                   query_pos_given_chain=(pos_chain_sum / np.maximum(chain_cnt, 1)).tolist(),
                                   query_pos_given_nonchain=(pos_non_sum / np.maximum(non_cnt, 1)).tolist())
    print(f"[{time.time()-t0:4.0f}s] {kind} L={L}: {B} words (model right on {p_ok:.3f})", flush=True)
    # a short printed summary: the heads with the largest next-hop / previous-hop shares, and the query's chain share
    for c in ("next", "prev"):
        ci = CATS.index(c)
        top = sorted(((word_mass[l, h, ci], l + 1, h) for l in range(NL) for h in range(H)), reverse=True)[:5]
        print(f"      top {c:4s}-hop shares (expected {word_exp[ci]:.3f}): " + "  ".join(f"L{l}H{h}={v:.2f}" for v, l, h in top), flush=True)
    top = sorted(((q_hop[l, h].sum(), l + 1, h) for l in range(NL) for h in range(H)), reverse=True)[:5]
    print(f"      query heads with most attention on hop letters (uniform {q_exp_chain:.2f}): " + "  ".join(f"L{l}H{h}={v:.2f}" for v, l, h in top), flush=True)
    json.dump(results, open(f"attention_{kind}{SUF}_results.json", "w"))
print("DONE", flush=True)
