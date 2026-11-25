"""
Test 7: linear probes at the INPUT (word) positions of torn-bigger-13.

Reading a word right to left, a pointer that leaves letter t at endpoint x continues through letters t-1, ..., 0 and ends at
pi_{t-1}(x), after H_{t-1}(x) more hops.  "Pointer jumping" (log-depth composition) would make letter t's state at layer l
carry pi_{t-1}(a) and pi_{t-1}(b) for chains of up to about 2**l remaining hops.  So for every layer l we fit ridge probes

        hidden state at word position t  ->  pi_{t-1}(a_t)   and   pi_{t-1}(b_t)        (25-way each)

on rows (word, t), and report accuracy by the number of remaining hops H.  (Input positions never see the outputs, so no
teacher forcing is involved.)  Data: planted words with L=12 and L=20 hops, and natural words over all 25 points.

Usage: python run_input_probes.py N_WORDS_PER_SET   (default 700 per set -> 2100 words, ~45k training rows)
"""
from paths import MODEL_TORN
import json, sys, time
import numpy as np
from np_transformer import NPTransformer
from permdata import plant_batch, apply_words, random_words

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 700
n, N = 25, 25
m = NPTransformer(MODEL_TORN)
rng = np.random.default_rng(777)
LAMS = [10.0, 100.0, 1000.0, 10000.0]


def prefix_tables(words):
    """PI[b, t, x] = pi_{t-1}(x) and HH[b, t, x] = remaining hops after leaving letter t at x (letters t-1..0)."""
    B = len(words)
    PI = np.zeros((B, N + 1, n), np.int64); HH = np.zeros((B, N + 1, n), np.int64)
    PI[:, 0] = np.arange(n)
    rows = np.arange(B)[:, None]
    for t in range(N):
        a, b = (words[:, t] // n)[:, None], (words[:, t] % n)[:, None]
        x = np.arange(n)[None]
        sx = np.where(a == b, x, np.where(x == a, b, np.where(x == b, a, x)))      # letter t sends x to sx
        PI[:, t + 1] = np.take_along_axis(PI[:, t], sx, 1)
        HH[:, t + 1] = np.take_along_axis(HH[:, t], sx, 1) + (sx != x)
    return PI, HH


def hidden_inputs(words, bs=50):
    """(B, 6, 25, D) residual stream at the 25 word positions after embedding (layer 0) and after each block."""
    perms = apply_words(words)
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
w, _ = plant_batch(rng, NW, 12, 9); sets.append(("L12", w))
w, _ = plant_batch(rng, NW, 20, 10); sets.append(("L20", w))
sets.append(("natural", random_words(rng, NW)))
words = np.concatenate([w for _, w in sets])
tag = np.concatenate([[nm] * NW for nm, _ in sets])
PI, HH = prefix_tables(words)
a_tok, b_tok = words // n, words % n
rowsA = np.arange(len(words))[:, None]
tA = np.arange(N)[None]
FA, FB = PI[rowsA, tA, a_tok], PI[rowsA, tA, b_tok]          # (B, N)
HA, HB = HH[rowsA, tA, a_tok], HH[rowsA, tA, b_tok]
H = hidden_inputs(words)
print(f"[{time.time()-t0:3.0f}s] hidden states {H.shape}", flush=True)

B = len(words)
rng2 = np.random.default_rng(5)
order = rng2.permutation(B)
tr, va, te = order[: int(.6 * B)], order[int(.6 * B): int(.8 * B)], order[int(.8 * B):]


is_tr = np.zeros(B, bool); is_tr[tr] = True
is_va = np.zeros(B, bool); is_va[va] = True
is_te = np.zeros(B, bool); is_te[te] = True


def probe_bin(l, F, Hh, lo, hi, shuffle=False):
    """Ridge probe trained ONLY on rows (word, t) whose remaining-hop count lies in [lo, hi]; returns (hits, n) on test rows."""
    inb = (Hh >= lo) & (Hh <= hi)
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


bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 24)]
res, ctrl = {}, {}
for l in range(m.n_blocks + 1):
    res[l] = {}
    for lo, hi in bins:
        hA, nA = probe_bin(l, FA, HA, lo, hi); hB, nB = probe_bin(l, FB, HB, lo, hi)
        res[l][f"{lo}-{hi}"] = [(hA + hB) / max(nA + nB, 1), nA + nB]
    print(f"[{time.time()-t0:3.0f}s] layer {l}: " + "  ".join(f"h{k}:{v[0]:.2f}" for k, v in res[l].items()), flush=True)
for l in (0, 4, 5):
    ctrl[l] = {}
    for lo, hi in ((1, 1), (5, 6), (13, 24)):
        hA, nA = probe_bin(l, FA, HA, lo, hi, True); hB, nB = probe_bin(l, FB, HB, lo, hi, True)
        ctrl[l][f"{lo}-{hi}"] = (hA + hB) / max(nA + nB, 1)
print("label-shuffle control:", ctrl, flush=True)
json.dump(dict(by_layer_and_remaining_hops=res, shuffle_control=ctrl, n_words=int(B), n_test_words=int(len(te)), chance=1 / 25),
          open("input_probe_results.json", "w"), indent=1)
print("DONE", flush=True)
