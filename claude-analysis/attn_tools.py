"""
Shared helpers for the attention tests (run_attention.py, run_head_ablation.py), for both models:

  "general": torn-bigger-13, 25 letters over 25 points, letter token = 25 a + b          (word positions 0..24)
  "elem":    elem-partitioned-long-boosted-1, 120 letters over 16 points, token t swaps t-1, t (word positions 0..119)

Planted words make slot k's pointer hop L times (general: over s = min(L+1, 10) points; elementary: over a window of width 8).
The position that predicts output slot k is N + k (N = number of word positions; START sits at N).
"""
import numpy as np
from np_transformer import NPTransformer, layer_norm, softmax
import ckpt_env

CONFIGS = {
    "general": dict(n=25, N=25, Ls=(8, 12, 16), model="torn-bigger-13"),
    "elem": dict(n=16, N=120, Ls=(12, 40, 80), model="elem-partitioned-long-boosted-1"),
}


def load(kind):
    """The model of this kind (see ckpt_env.py for running on another checkpoint) and its config."""
    c = dict(CONFIGS[kind])
    if ckpt_env.LS:
        c["Ls"] = ckpt_env.LS
    path = f"../scaling-generator/trained-models/{c['model']}/model.safetensors"
    if kind == "general":
        from paths import MODEL_TORN
        m = NPTransformer(ckpt_env.CKPT or MODEL_TORN, n_blocks=ckpt_env.BLOCKS)
    else:
        m = NPTransformer(ckpt_env.CKPT or path, max_group=c["n"], input_len=c["N"], num_trans=c["n"], n_blocks=ckpt_env.BLOCKS)
    return m, c


def endpoints(kind, words):
    """(EA, EB) endpoint arrays of every letter; -1 for identity letters.  Shapes like words."""
    if kind == "general":
        a, b = words // 25, words % 25
        ident = a == b
        return np.where(ident, -1, a), np.where(ident, -1, b)
    ident = words == 0
    return np.where(ident, -1, words - 1), np.where(ident, -1, words)


def hop_letters(kind, word, k):
    """[(t, before, after)] for the hop letters of slot k, in reading order (right to left)."""
    ea, eb = endpoints(kind, np.asarray(word))
    x, out = int(k), []
    for t in range(len(word) - 1, -1, -1):
        if ea[t] >= 0 and x in (ea[t], eb[t]):
            y = int(eb[t] if x == ea[t] else ea[t])
            out.append((t, x, y)); x = y
    return out


def apply(kind, words):
    if kind == "general":
        from permdata import apply_words
        return apply_words(words)
    from elem_tools import apply_words_elem
    return apply_words_elem(words)


def plant(kind, rng, B, L):
    if kind == "general":
        from permdata import plant_batch
        return plant_batch(rng, B, L, min(L + 1, 10))
    from elem_tools import plant_elem_batch
    return plant_elem_batch(rng, B, L, 8)


def make_toks(m, words, perms):
    return np.concatenate([words, np.full((len(words), 1), m.START, words.dtype), perms[:, :-1] + m.num_trans], 1)


def predict_slot(m, toks, ks, bs=50):
    """Teacher-forced prediction (permutation entry) at slot ks[i], for each row."""
    out = np.zeros(len(toks), int)
    for s in range(0, len(toks), bs):
        lg = m.forward(toks[s:s + bs])
        sl = slice(s, s + bs)
        out[sl] = lg[np.arange(len(lg)), ks[sl]].argmax(-1) - m.num_trans
    return out


def planted_words(kind, m, rng, B, L):
    """Planted words with L hops, kept only if the model answers the planted slot correctly (teacher forced)."""
    words, ks = plant(kind, rng, B, L)
    perm = apply(kind, words)
    ok = predict_slot(m, make_toks(m, words, perm), ks) == perm[np.arange(B), ks]
    return words[ok], ks[ok], perm[ok], float(ok.mean())


def forward_ablate(m, toks, head_off, collect=False):
    """Same maths as m.forward, with one attention head's output zeroed at some positions.
    head_off = None or (layer, head, boolean mask over positions (T,)) or (layer, head, mask, repl) with repl of shape (T, hd):
    the head's output is then replaced by repl (a mean over other words) instead of zero.
    collect=True also returns the per-layer head outputs (B, T, H, hd) of the unablated pass.
    Returns logits of the last max_group positions."""
    outs = []
    B, T = toks.shape
    x = m.E[toks] + m.P[:T][None]
    M = m.mask(T)
    for li, ly in enumerate(m.layers):
        a = layer_norm(x, ly["ln1w"], ly["ln1b"])
        q = (a @ ly["Wq"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        k = (a @ ly["Wk"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        v = (a @ ly["Wv"].T).reshape(B, T, m.n_head, m.hd).transpose(0, 2, 1, 3)
        s = np.where(M[None, None] == 0, -np.inf, (q @ k.transpose(0, 1, 3, 2)) * m.scale)
        o = (softmax(s) @ v).transpose(0, 2, 1, 3)                 # (B, T, H, hd)
        if collect: outs.append(o.copy())
        if head_off is not None and head_off[0] == li:
            o = o.copy()
            if len(head_off) == 4:
                o[:, head_off[2], head_off[1], :] = head_off[3][head_off[2]][None]
            else:
                o[:, head_off[2], head_off[1], :] = 0.0
        o = o.reshape(B, T, m.D)
        x = x + (o @ ly["Wo"].T + ly["bo"])
        f = layer_norm(x, ly["ln2w"], ly["ln2b"])
        x = x + (np.maximum(f @ ly["W1"].T + ly["b1"], 0) @ ly["W2"].T + ly["b2"])
    lg = layer_norm(x, m.lnfw, m.lnfb)[:, -m.max_group:, :] @ m.Wlm.T
    return (lg, outs) if collect else lg
