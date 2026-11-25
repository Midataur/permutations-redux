"""
Test 8 for the ELEMENTARY model (elem-partitioned-long-boosted-1): activation patching at the word positions.
Same design as run_patching.py (general model, section 3.10); differences forced by the elementary letters:

  * a letter is a token t >= 1 that swaps positions t-1 and t (token 0 = identity), so a hop moves the pointer by +-1;
  * the j-th hop letter (x_{j-1} -> x_j) is corrupted into the letter that moves the pointer to the OTHER neighbour,
    z = 2 x_{j-1} - x_j.  If z is outside 0..15 the word is skipped (no such letter exists);
  * after the corruption the pointer follows its own chain of hop letters, so besides the clean chain I also patch the
    hop letters of the corrupted pointer ("corrchain"), and "nonchain" excludes both chains;
  * the word has 120 letters; planted words make L hops over a window of width w = 8 (elem_tools.plant_elem_batch).

Teacher forcing uses the CLEAN word's outputs in every run.  Patching at layer l (0 = embeddings, l = after block l)
overwrites the residual stream at a set of word positions by the clean run's values.  The patched run only has to
recompute blocks l+1..5 (blocks 1..l are identical to the corrupted run), which is mathematically the same as
run_patching.py but about twice as fast.  Patching at layer 5 changes nothing (no later block reads the word positions),
so layer 5 is reported as the unpatched baseline.

Usage: python run_patching_elem.py N_WORDS_PER_CELL
"""
import json, sys, time
import numpy as np
from np_transformer import NPTransformer, layer_norm, softmax
from elem_tools import *

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MODEL = "../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
m = NPTransformer(MODEL, max_group=G, input_len=N, num_trans=G)
rng = np.random.default_rng(2025)
W_WIDTH = 8
KINDS = ("all", "chain", "corrchain", "left", "right", "self", "nonchain")


def hop_letters(word, k):
    """[(t, before, after)] for the hop letters of slot k, in reading order (right to left)."""
    x, out = int(k), []
    for t in range(N - 1, -1, -1):
        tok = int(word[t])
        if tok > 0 and x in (tok - 1, tok):
            y = tok if x == tok - 1 else tok - 1
            out.append((t, x, y)); x = y
    return out


def make_toks(words, perms_prefix):
    B = len(words)
    return np.concatenate([words, np.full((B, 1), m.START, words.dtype), perms_prefix[:, :-1] + m.num_trans], 1)


def block(x, ly, Mm):
    B, T, _ = x.shape
    a_ = layer_norm(x, ly["ln1w"], ly["ln1b"])
    q = (a_ @ ly["Wq"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
    k_ = (a_ @ ly["Wk"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
    v = (a_ @ ly["Wv"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
    sc = np.where(Mm[None, None] == 0, -np.inf, (q @ k_.transpose(0, 1, 3, 2)) * m.scale)
    o = (softmax(sc) @ v).transpose(0, 2, 1, 3).reshape(B, T, m.D)
    x = x + (o @ ly["Wo"].T + ly["bo"])
    f = layer_norm(x, ly["ln2w"], ly["ln2b"])
    return x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])


def run_from(x, l, Mm):
    """Residual stream x at layer l -> all hidden states h[l..5] (list indexed from l)."""
    hs = [x]
    for li in range(l, m.n_blocks):
        x = block(x, m.layers[li], Mm); hs.append(x)
    return hs


def answer(x, ks):
    """Prediction (permutation entry) at output slot ks[i] for each word, from the final residual stream x."""
    T = x.shape[1]
    sel = x[np.arange(len(x)), T - m.max_group + ks]
    return (layer_norm(sel, m.lnfw, m.lnfb) @ m.Wlm.T).argmax(-1) - m.num_trans


def embed(toks):
    return m.E[toks] + m.P[:toks.shape[1]][None]


def run_cell(words, ks, hops, j, bs=50):
    B = len(words)
    clean_perm = apply_words_elem(words)
    cw = words.copy()
    valid = np.ones(B, bool)
    chain_pos, chain_pos_c, tj = [], [], np.zeros(B, int)
    for i in range(B):
        t, xb, xa = hops[i][j]
        z = 2 * xb - xa
        if not 0 <= z < G:
            valid[i] = False; chain_pos.append([]); chain_pos_c.append([]); continue
        cw[i, t] = max(xb, z)
        tj[i] = t
        chain_pos.append([h[0] for h in hops[i]])
        chain_pos_c.append([h[0] for h in hop_letters(cw[i], ks[i])])      # hop letters of the CORRUPTED pointer
    cperm = apply_words_elem(cw)
    ar = np.arange(B)
    differs = (cperm[ar, ks] != clean_perm[ar, ks]) & valid
    toks_clean, toks_corr = make_toks(words, clean_perm), make_toks(cw, clean_perm)   # corrupted word, CLEAN prefix
    T = toks_clean.shape[1]
    Mm = m.mask(T)

    def mask_for(kind):
        M = np.zeros((B, T), bool)
        for i in range(B):
            t = tj[i]
            if kind == "all": M[i, :N] = True
            elif kind == "chain": M[i, chain_pos[i]] = True
            elif kind == "left": M[i, :t] = True
            elif kind == "right": M[i, t + 1:N] = True
            elif kind == "corrchain": M[i, chain_pos_c[i]] = True
            elif kind == "self": M[i, t] = True
            elif kind == "nonchain":
                M[i, :N] = True; M[i, chain_pos[i]] = False; M[i, chain_pos_c[i]] = False
        return M
    masks = {kd: mask_for(kd) for kd in KINDS}
    out = {kd: np.zeros((6, B, 2), bool) for kd in KINDS}        # [layer, word, (restored, follows_corrupted)]
    base = np.zeros((B, 2), bool)
    for s in range(0, B, bs):
        sl = slice(s, s + bs)
        k_ = ks[sl]; p_clean, p_corr = clean_perm[sl][np.arange(len(k_)), k_], cperm[sl][np.arange(len(k_)), k_]
        h_c = run_from(embed(toks_clean[sl]), 0, Mm)
        h_k = run_from(embed(toks_corr[sl]), 0, Mm)
        pr = answer(h_k[5], k_)
        base[sl, 0], base[sl, 1] = pr == p_clean, pr == p_corr
        for kd in KINDS:
            msk = masks[kd][sl][:, :, None]
            out[kd][5, sl, 0], out[kd][5, sl, 1] = base[sl, 0], base[sl, 1]      # nothing reads the word positions after block 5
            for l in range(5):
                x = np.where(msk, h_c[l], h_k[l])
                pr = answer(run_from(x, l, Mm)[-1], k_)
                out[kd][l, sl, 0], out[kd][l, sl, 1] = pr == p_clean, pr == p_corr
    keep = differs
    return dict(n=int(keep.sum()), n_skipped_no_other_neighbour=int((~valid).sum()),
                unpatched=[float(base[keep, 0].mean()), float(base[keep, 1].mean())],
                patched={kd: [[float(out[kd][l, keep, 0].mean()), float(out[kd][l, keep, 1].mean())] for l in range(6)] for kd in out})


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    for L in (12, 24, 40, 80):
        words, ks = plant_elem_batch(rng, NW, L, W_WIDTH)
        hops = [hop_letters(w, k) for w, k in zip(words, ks)]
        assert all(len(h) == L for h in hops)
        perm = apply_words_elem(words)
        ok = np.zeros(NW, bool)
        for s in range(0, NW, 50):
            lg = m.forward(make_toks(words[s:s + 50], perm[s:s + 50]))
            ok[s:s + 50] = lg[np.arange(len(lg)), ks[s:s + 50]].argmax(-1) - m.num_trans == perm[s:s + 50][np.arange(len(lg)), ks[s:s + 50]]
        words, ks, hops = words[ok], ks[ok], [h for h, o in zip(hops, ok) if o]
        print(f"[{time.time()-t0:5.0f}s] L={L} w={W_WIDTH}: model right on {ok.mean():.3f} of the planted words; using {len(words)}", flush=True)
        for jname, j in (("first", 0), ("middle", L // 2), ("last", L - 1)):
            r = run_cell(words, ks, hops, j)
            results[f"L{L}_{jname}"] = r
            pa = r["patched"]
            print(f"[{time.time()-t0:5.0f}s] L={L} corrupt hop #{j+1} ({jname}), n={r['n']} (skipped {r['n_skipped_no_other_neighbour']}): "
                  f"unpatched restored/follows-corrupt = {r['unpatched'][0]:.2f}/{r['unpatched'][1]:.2f}", flush=True)
            for kd in pa:
                print("     patch %-9s restored by layer: " % kd + " ".join(f"{pa[kd][l][0]:.2f}" for l in range(6)), flush=True)
            json.dump(results, open("patching_elem_results.json", "w"), indent=1)
    print("DONE", flush=True)
