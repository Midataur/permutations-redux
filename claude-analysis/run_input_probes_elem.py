"""
Test 7 for the ELEMENTARY model (elem-partitioned-long-boosted-1): linear probes at the INPUT (word) positions.
Same design as run_input_probes.py (general model, section 3.9).

Reading a word right to left, a pointer that leaves letter t at endpoint x continues through letters t-1, ..., 0 and ends at
pi_{t-1}(x) after H_{t-1}(x) more hops.  Letter t is token tok >= 1 with endpoints a = tok-1 and b = tok (token 0 is the
identity and has no endpoints, so those rows are left out).  For every layer l (0 = embeddings, l = after block l) a ridge probe
maps the hidden state at word position t to pi_{t-1}(a_t) and pi_{t-1}(b_t) (16 classes each), trained and tested separately
for each range of H (remaining hops).  Input positions never see the outputs, so no teacher forcing is involved.

Data: natural words (120 uniform letters over tokens 0..15, as in the real test file) and planted words with L = 40 and
L = 80 hops over a window of width 8, which supply the long remaining-hop ranges.  Words are split 60/20/20 into train,
validation (ridge strength) and test, so no word appears in two parts.

Usage: python run_input_probes_elem.py N_WORDS_PER_SET   (default 450 per set -> 1350 words)
"""
import json, sys, time
import numpy as np
from np_transformer import NPTransformer
from elem_tools import *

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 450
MODEL = "../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors"
m = NPTransformer(MODEL, max_group=G, input_len=N, num_trans=G)
rng = np.random.default_rng(778)
LAMS = [10.0, 100.0, 1000.0, 10000.0]
n = G


def prefix_tables(words):
    """PI[b, t, x] = pi_{t-1}(x) and HH[b, t, x] = remaining hops after leaving letter t at x (letters t-1..0)."""
    B = len(words)
    PI = np.zeros((B, N + 1, n), np.int64); HH = np.zeros((B, N + 1, n), np.int64)
    PI[:, 0] = np.arange(n)
    x = np.arange(n)[None]
    for t in range(N):
        tok = words[:, t][:, None]
        a, b = tok - 1, tok
        sx = np.where(tok == 0, x, np.where(x == a, b, np.where(x == b, a, x)))      # letter t sends x to sx
        PI[:, t + 1] = np.take_along_axis(PI[:, t], sx, 1)
        HH[:, t + 1] = np.take_along_axis(HH[:, t], sx, 1) + (sx != x)
    return PI, HH


def hidden_inputs(words, bs=25):
    """(B, 6, 120, D) residual stream at the word positions after embedding (layer 0) and after each block."""
    perms = apply_words_elem(words)
    out = np.zeros((len(words), m.n_blocks + 1, N, m.D), np.float32)
    for s in range(0, len(words), bs):
        w, p = words[s:s + bs], perms[s:s + bs]
        toks = np.concatenate([w, np.full((len(w), 1), m.START, w.dtype), p[:, :-1] + m.num_trans], 1)
        _, hid = m.forward(toks, return_hidden=True)
        for l in range(m.n_blocks + 1):
            out[s:s + len(w), l] = hid[l][:, :N]
    return out


t0 = time.time()
sets = []
w, _ = plant_elem_batch(rng, NW, 40, 8); sets.append(("L40", w))
w, _ = plant_elem_batch(rng, NW, 80, 8); sets.append(("L80", w))
sets.append(("natural", rng.integers(0, G, size=(NW, N))))
words = np.concatenate([w for _, w in sets])
src_of_word = np.concatenate([[nm] * NW for nm, _ in sets])
PI, HH = prefix_tables(words)
tok = words
a_tok, b_tok = np.maximum(tok - 1, 0), tok                      # endpoints (rows with tok == 0 are excluded below)
rowsA = np.arange(len(words))[:, None]
tA = np.arange(N)[None]
FA, FB = PI[rowsA, tA, a_tok], PI[rowsA, tA, b_tok]             # (B, N)
HA, HB = HH[rowsA, tA, a_tok], HH[rowsA, tA, b_tok]
real = tok > 0                                                  # rows that are real transpositions
H = hidden_inputs(words)
print(f"[{time.time()-t0:3.0f}s] hidden states {H.shape}; real-letter rows {int(real.sum())} of {real.size}", flush=True)

B = len(words)
rng2 = np.random.default_rng(5)
order = rng2.permutation(B)
tr, va, te = order[: int(.6 * B)], order[int(.6 * B): int(.8 * B)], order[int(.8 * B):]
is_tr = np.zeros(B, bool); is_tr[tr] = True
is_va = np.zeros(B, bool); is_va[va] = True
is_te = np.zeros(B, bool); is_te[te] = True


def probe_bin(l, F, Hh, lo, hi, shuffle=False, src=None, trange=None):
    """Ridge probe trained ONLY on rows (word, t) whose remaining-hop count lies in [lo, hi]; returns (hits, n) on test rows."""
    inb = (Hh >= lo) & (Hh <= hi) & real
    if src is not None:
        inb = inb & (src_of_word == src)[:, None]
    if trange is not None:
        inb = inb & (tA >= trange[0]) & (tA < trange[1])
    def sel(mask):
        wi, ti = np.nonzero(inb & mask[:, None])
        return H[wi, l, ti], F[wi, ti]
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = sel(is_tr), sel(is_va), sel(is_te)
    if len(Ytr) < 200 or len(Yte) < 50:
        return 0, 0
    if shuffle:
        Ytr = Ytr[rng2.permutation(len(Ytr))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    lam, Q = np.linalg.eigh(Xtr.T @ Xtr)
    XtY = Q.T @ (Xtr.T @ np.eye(n)[Ytr])
    best = (-1, None)
    for lm in LAMS:
        Wm = Q @ (XtY / (lam + lm)[:, None])
        acc = (np.argmax(Xva @ Wm, 1) == Yva).mean()
        if acc > best[0]:
            best = (acc, Wm)
    return int((np.argmax(Xte @ best[1], 1) == Yte).sum()), len(Yte)


bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 24), (25, 48), (49, 120)]
res, ctrl = {}, {}
for l in range(m.n_blocks + 1):
    res[l] = {}
    for lo, hi in bins:
        hA, nA = probe_bin(l, FA, HA, lo, hi); hB, nB = probe_bin(l, FB, HB, lo, hi)
        res[l][f"{lo}-{hi}"] = [(hA + hB) / max(nA + nB, 1), nA + nB]
    print(f"[{time.time()-t0:3.0f}s] layer {l}: " + "  ".join(f"h{k}:{v[0]:.2f}" for k, v in res[l].items()), flush=True)
    json.dump(dict(by_layer_and_remaining_hops=res, n_words=int(B), n_test_words=int(len(te)), chance=1 / n),
              open("input_probe_elem_results.json", "w"), indent=1)
for l in (0, 4, 5):
    ctrl[l] = {}
    for lo, hi in ((1, 1), (5, 6), (13, 24), (25, 48)):
        hA, nA = probe_bin(l, FA, HA, lo, hi, True); hB, nB = probe_bin(l, FB, HB, lo, hi, True)
        ctrl[l][f"{lo}-{hi}"] = (hA + hB) / max(nA + nB, 1)
print("label-shuffle control:", ctrl, flush=True)

# Second phase: the same probes trained and tested within one data source only, with finer bins at long range.
bins2 = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 16), (17, 24), (25, 32), (33, 48), (49, 120)]
by_src = {}
for sname in ("natural", "L40", "L80"):
    by_src[sname] = {}
    for l in range(m.n_blocks + 1):
        by_src[sname][l] = {}
        for lo, hi in bins2:
            hA, nA = probe_bin(l, FA, HA, lo, hi, src=sname); hB, nB = probe_bin(l, FB, HB, lo, hi, src=sname)
            by_src[sname][l][f"{lo}-{hi}"] = [(hA + hB) / max(nA + nB, 1), nA + nB]
        print(f"[{time.time()-t0:3.0f}s] {sname:8s} layer {l}: " + "  ".join(f"h{k}:{v[0]:.2f}({v[1]})" for k, v in by_src[sname][l].items() if v[1]), flush=True)

# Third phase: does long-range decodability depend on the position t in the word rather than on the number of hops?
by_t = {}
tgroups = [(0, 60), (60, 100), (100, 120)]
bins3 = [(5, 6), (7, 8), (9, 12), (13, 16), (17, 24), (25, 48), (49, 120)]
for sname in ("natural", "L80"):
    by_t[sname] = {}
    for l in (0, 4, 5):
        by_t[sname][l] = {}
        for (t0_, t1_) in tgroups:
            by_t[sname][l][f"t{t0_}-{t1_}"] = {}
            for lo, hi in bins3:
                hA, nA = probe_bin(l, FA, HA, lo, hi, src=sname, trange=(t0_, t1_)); hB, nB = probe_bin(l, FB, HB, lo, hi, src=sname, trange=(t0_, t1_))
                by_t[sname][l][f"t{t0_}-{t1_}"][f"{lo}-{hi}"] = [(hA + hB) / max(nA + nB, 1), nA + nB]
            row = by_t[sname][l][f"t{t0_}-{t1_}"]
            print(f"[{time.time()-t0:3.0f}s] {sname:8s} layer {l} t in [{t0_},{t1_}): " + "  ".join(f"h{k}:{v[0]:.2f}({v[1]})" for k, v in row.items() if v[1]), flush=True)
json.dump(dict(by_layer_and_remaining_hops=res, by_source=by_src, by_position=by_t, shuffle_control=ctrl, n_words=int(B), n_test_words=int(len(te)), chance=1 / n),
          open("input_probe_elem_results.json", "w"), indent=1)
print("DONE", flush=True)
