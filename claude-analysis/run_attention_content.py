"""
Test 11b: is the attention to the next / previous hop letter about content, or only about position?

In planted words most letters are chain letters, so a head that just looks at the nearest letters to the left would look like a
"next hop" head.  Here, on NATURAL words (where a chain is a small part of the word), for every hop letter at word position t
and every offset d in -6..-1, +1..+6 (d < 0: to the left, read later; d > 0: to the right, read earlier), compare
  attention to the letter at t+d when it IS the next hop letter (d < 0) or the previous hop letter (d > 0)
  with   attention to the letter at t+d when it is NOT on the chain at all,
averaged over all (slot, hop letter) pairs with at least 2 hops.  A head that reads content has a hop value well above the
non-chain value at the same offset; a purely positional head has them equal.

Usage: python run_attention_content.py MODEL N_WORDS      -> attention_content_<MODEL>_results.json
"""
import json, sys, time
import numpy as np
from attn_tools import *
from ckpt_env import SUF

kind = sys.argv[1]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 200
m, cfg = load(kind)
N, H, NL, G = cfg["N"], m.n_head, m.n_blocks, m.max_group
rng = np.random.default_rng({"general": 61, "elem": 62}[kind])
if kind == "general":
    from permdata import random_words
    words = random_words(rng, NW)
else:
    words = rng.integers(0, 16, size=(NW, N))
perm = apply(kind, words)
toks = make_toks(m, words, perm)
OFFS = [d for d in range(-6, 7) if d != 0]
hop_sum = np.zeros((NL, H, len(OFFS))); hop_n = np.zeros(len(OFFS))
non_sum = np.zeros((NL, H, len(OFFS))); non_n = np.zeros(len(OFFS))
dec_sum = np.zeros((NL, H, len(OFFS))); dec_n = np.zeros(len(OFFS))     # decoys: non-chain letters holding the exit point (d<0) / the entry point (d>0)
t0 = time.time(); npairs = 0
for s in range(0, NW, 25):
    _, attns = m.forward(toks[s:s + 25], return_attn=True)
    for bi in range(len(toks[s:s + 25])):
        i = s + bi
        EA, EB = endpoints(kind, words[i])
        for k in range(G):
            hp = hop_letters(kind, words[i], k)
            if len(hp) < 2: continue
            ts = [h[0] for h in hp]; chain = set(ts)
            for j, t in enumerate(ts):
                nxt = ts[j + 1] if j + 1 < len(ts) else None
                prv = ts[j - 1] if j >= 1 else None
                npairs += 1
                for di, d in enumerate(OFFS):
                    u = t + d
                    if not (0 <= u < N): continue
                    if (d < 0 and u == nxt) or (d > 0 and u == prv):
                        for l in range(NL): hop_sum[l, :, di] += attns[l][bi][:, t, u]
                        hop_n[di] += 1
                    elif u not in chain:
                        for l in range(NL): non_sum[l, :, di] += attns[l][bi][:, t, u]
                        non_n[di] += 1
                        pt = hp[j][2] if d < 0 else hp[j][1]          # exit point (letters to the left) / entry point (to the right)
                        if EA[u] == pt or EB[u] == pt:
                            for l in range(NL): dec_sum[l, :, di] += attns[l][bi][:, t, u]
                            dec_n[di] += 1
hop = hop_sum / np.maximum(hop_n, 1); non = non_sum / np.maximum(non_n, 1); dec = dec_sum / np.maximum(dec_n, 1)
res = dict(model=cfg["model"], N=N, offsets=OFFS, n_words=NW, n_pairs=npairs, hop_n=hop_n.tolist(), non_n=non_n.tolist(),
           attn_to_hop=hop.tolist(), attn_to_nonchain=non.tolist(), attn_to_decoy=dec.tolist(), decoy_n=dec_n.tolist())
json.dump(res, open(f"attention_content_{kind}{SUF}_results.json", "w"))
print(f"[{time.time()-t0:.0f}s] {kind}: {npairs} (slot, hop letter) pairs with >=2 hops; hop-letter cases per offset: {hop_n.astype(int).tolist()}; decoy cases: {dec_n.astype(int).tolist()}", flush=True)
print("head   offset:  " + " ".join(f"{d:6d}" for d in OFFS))
for l in range(NL):
    for h in range(H):
        r = hop[l, h] / np.maximum(non[l, h], 1e-9)
        if r.max() > 4:
            print(f"blk{l+1} h{h} ratio:  " + " ".join(f"{x:6.1f}" for x in r) + f"   (hop attention x uniform at best offset: {hop[l,h].max()*N:.1f})")
print("DONE")
