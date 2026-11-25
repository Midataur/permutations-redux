"""
Test 8: activation patching and corruption influence at the INPUT positions of torn-bigger-13.

A planted word w (the pointer of slot k makes L hops; the model is right on w) is corrupted at one hop letter: the j-th hop
letter (x_{j-1}, x_j) is replaced by (x_{j-1}, z) with z a fresh point, so the pointer leaves the chain there and the true
answer for slot k becomes p*_k != p_k.  Teacher forcing uses the CLEAN word's outputs as the prefix in every run.

(1) Influence: relative change of the residual stream at every word position and layer, clean vs corrupted run.
(2) Patching: run the corrupted word, but at layer l (0 = embeddings, l = after block l) overwrite the residual stream at a SET
    of word positions by the clean run's values.  We report the fraction of words whose prediction at slot k goes back to the
    clean answer p_k ("restored") and the fraction that still follows the corrupted word p*_k.
    Position sets: all 25 word positions; the chain's hop letters; the letters to the LEFT of the corrupted one (read later);
    the letters to the RIGHT of it (read earlier); the corrupted letter only; the non-chain letters (control).

Usage: python run_patching.py N_WORDS
"""
from paths import MODEL_TORN
import json, sys, time
import numpy as np
from np_transformer import NPTransformer, layer_norm, softmax
from permdata import plant_batch, apply_words

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 200
n, N = 25, 25
m = NPTransformer(MODEL_TORN)
rng = np.random.default_rng(2024)


def forward_patched(toks, patch=None):
    """Same maths as m.forward; patch = {layer: (mask (T,) bool, clean (B,T,D))}.  Returns logits and hidden states."""
    B, T = toks.shape
    x = m.E[toks] + m.P[:T][None]
    M = m.mask(T)
    hidden = []
    def apply(l, x):
        if patch and l in patch:
            msk, clean = patch[l]
            x = x.copy(); x[:, msk] = clean[:, msk]
        return x
    x = apply(0, x); hidden.append(x)
    for li, ly in enumerate(m.layers):
        a = layer_norm(x, ly["ln1w"], ly["ln1b"])
        q = (a @ ly["Wq"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        k = (a @ ly["Wk"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        v = (a @ ly["Wv"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        s = np.where(M[None, None] == 0, -np.inf, (q @ k.transpose(0, 1, 3, 2)) * m.scale)
        o = (softmax(s) @ v).transpose(0, 2, 1, 3).reshape(B, T, m.D)
        x = x + (o @ ly["Wo"].T + ly["bo"])
        f = layer_norm(x, ly["ln2w"], ly["ln2b"])
        x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
        x = apply(li + 1, x); hidden.append(x)
    xf = layer_norm(x, m.lnfw, m.lnfb)
    return xf[:, -m.max_group:, :] @ m.Wlm.T, hidden


def hop_letters(word, k):
    """[(t, before, after)] for the hop letters of slot k, in reading order."""
    x, out = int(k), []
    for t in range(N - 1, -1, -1):
        a, b = int(word[t]) // n, int(word[t]) % n
        if a != b and x in (a, b):
            y = b if x == a else a
            out.append((t, x, y)); x = y
    return out


def make_toks(words, perms_prefix):
    B = len(words)
    return np.concatenate([words, np.full((B, 1), m.START, words.dtype), perms_prefix[:, :-1] + m.num_trans], 1)


def run_batch(words, ks, hops, j, bs=100):
    """Corrupt the j-th hop letter of every word; return dict of results."""
    B = len(words)
    clean_perm = apply_words(words)
    cw = words.copy()
    corrupted_ok = np.zeros(B, bool)
    chain_pos, tj = [], np.zeros(B, int)
    for i in range(B):
        hl = hops[i]
        t, xb, xa = hl[j]
        path = {h[1] for h in hl} | {h[2] for h in hl}
        z = int(rng.choice([p for p in range(n) if p not in path]))
        a, b = (xb, z) if rng.random() < 0.5 else (z, xb)
        cw[i, t] = a * n + b
        tj[i] = t
        chain_pos.append([h[0] for h in hl])
    cperm = apply_words(cw)
    ar = np.arange(B)
    differs = cperm[ar, ks] != clean_perm[ar, ks]
    toks_clean = make_toks(words, clean_perm)
    toks_corr = make_toks(cw, clean_perm)                  # corrupted word, CLEAN output prefix
    res = {}
    # sets of positions per word differ, so build masks per word
    T = 50
    def mask_for(kind):
        M = np.zeros((B, T), bool)
        for i in range(B):
            t = tj[i]
            if kind == "all": M[i, :N] = True
            elif kind == "chain": M[i, chain_pos[i]] = True
            elif kind == "left": M[i, :t] = True
            elif kind == "right": M[i, t + 1:N] = True
            elif kind == "self": M[i, t] = True
            elif kind == "nonchain":
                M[i, :N] = True; M[i, chain_pos[i]] = False
        return M
    masks = {kd: mask_for(kd) for kd in ("all", "chain", "left", "right", "self", "nonchain")}
    out = {kd: np.zeros((6, B, 2), bool) for kd in masks}           # [layer, word, (restored, follows_corrupted)]
    base = np.zeros((B, 2), bool); infl = np.zeros((6, N), np.float64)
    for s in range(0, B, bs):
        sl = slice(s, s + bs)
        lg_c, h_c = forward_patched(toks_clean[sl])
        lg_k, h_k = forward_patched(toks_corr[sl])
        idx = np.arange(len(words[sl]))
        base[sl, 0] = lg_k[idx, ks[sl]].argmax(-1) - m.num_trans == clean_perm[sl][idx, ks[sl]]
        base[sl, 1] = lg_k[idx, ks[sl]].argmax(-1) - m.num_trans == cperm[sl][idx, ks[sl]]
        for l in range(6):
            d = np.linalg.norm(h_k[l][:, :N] - h_c[l][:, :N], axis=-1) / np.linalg.norm(h_c[l][:, :N], axis=-1)
            # align by position relative to the corrupted letter: store the mean per absolute position (grouped later)
            infl[l] += d.sum(0)
        for kd, M in masks.items():
            for l in range(6):
                msk_b = M[sl]
                # patch with per-word masks: apply word by word through a combined clean tensor
                # implement via blending: build patched hidden at layer l = where(mask, clean, corrupted) inside forward
                pl = {l: (None, None)}
                def fwd_blend(toks, l=l, msk_b=msk_b, h_clean=h_c):
                    Bq, Tq = toks.shape
                    x = m.E[toks] + m.P[:Tq][None]
                    Mm = m.mask(Tq)
                    def blend(li, x):
                        if li == l:
                            x = np.where(msk_b[:, :, None], h_clean[li], x)
                        return x
                    x = blend(0, x)
                    for li, ly in enumerate(m.layers):
                        a_ = layer_norm(x, ly["ln1w"], ly["ln1b"])
                        q = (a_ @ ly["Wq"].T).reshape(Bq, Tq, m.n_head, m.hd).transpose(0, 2, 1, 3)
                        k_ = (a_ @ ly["Wk"].T).reshape(Bq, Tq, m.n_head, m.hd).transpose(0, 2, 1, 3)
                        v = (a_ @ ly["Wv"].T).reshape(Bq, Tq, m.n_head, m.hd).transpose(0, 2, 1, 3)
                        sc = np.where(Mm[None, None] == 0, -np.inf, (q @ k_.transpose(0, 1, 3, 2)) * m.scale)
                        o = (softmax(sc) @ v).transpose(0, 2, 1, 3).reshape(Bq, Tq, m.D)
                        x = x + (o @ ly["Wo"].T + ly["bo"])
                        f = layer_norm(x, ly["ln2w"], ly["ln2b"])
                        x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
                        x = blend(li + 1, x)
                    return layer_norm(x, m.lnfw, m.lnfb)[:, -m.max_group:, :] @ m.Wlm.T
                lg = fwd_blend(toks_corr[sl])
                pr = lg[idx, ks[sl]].argmax(-1) - m.num_trans
                out[kd][l, sl, 0] = pr == clean_perm[sl][idx, ks[sl]]
                out[kd][l, sl, 1] = pr == cperm[sl][idx, ks[sl]]
    keep = differs
    res = dict(n=int(keep.sum()),
               unpatched=[float(base[keep, 0].mean()), float(base[keep, 1].mean())],
               patched={kd: [[float(out[kd][l, keep, 0].mean()), float(out[kd][l, keep, 1].mean())] for l in range(6)] for kd in out},
               influence_by_position=(infl / B).tolist())
    # influence relative to the corrupted letter's position (offset = t - t_j), averaged
    return res


t0 = time.time()
results = {}
for L in (8, 12, 16):
    s_ = min(L + 1, 10)
    words, ks = plant_batch(rng, NW, L, s_)
    hops = [hop_letters(w, k) for w, k in zip(words, ks)]
    assert all(len(h) == L for h in hops)
    # keep only words the model gets right (clean) at the planted slot
    perm = apply_words(words)
    lg, _ = forward_patched(make_toks(words, perm))
    ok = lg[np.arange(NW), ks].argmax(-1) - m.num_trans == perm[np.arange(NW), ks]
    words, ks, hops = words[ok], ks[ok], [h for h, o in zip(hops, ok) if o]
    print(f"[{time.time()-t0:4.0f}s] L={L} s={s_}: model right on {ok.mean():.3f} of the planted words; using {len(words)}", flush=True)
    for jname, j in (("first", 0), ("middle", L // 2), ("last", L - 1)):
        r = run_batch(words, ks, hops, j)
        results[f"L{L}_{jname}"] = r
        pa = r["patched"]
        print(f"[{time.time()-t0:4.0f}s] L={L} corrupt hop #{j+1} ({jname}), n={r['n']}: unpatched restored/follows-corrupt = {r['unpatched'][0]:.2f}/{r['unpatched'][1]:.2f}", flush=True)
        for kd in pa:
            print("     patch %-9s restored by layer: " % kd + " ".join(f"{pa[kd][l][0]:.2f}" for l in range(6)), flush=True)
json.dump(results, open("patching_results.json", "w"), indent=1)
print("DONE", flush=True)
