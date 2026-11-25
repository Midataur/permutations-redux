"""
Test 1: layer-wise linear probes for the pointer's position after r hops, torn-bigger-13 (general transpositions).

For a planted word (the pointer of slot k makes exactly L hops and visits s distinct points) we run the model with
teacher forcing and take the residual-stream vector at the position that predicts p_k (position 25+k) after the
embedding (layer 0) and after each of the 5 blocks (layers 1..5). For every layer l and hop index r = 0..L we fit a
ridge-regression probe  h^(l) -> one-hot(x_r)  (x_r = pointer position after r hops; x_0 = k, x_L = p_k) on 60% of
the words, choose the ridge strength on 20%, and report 25-way accuracy on the last 20% (chance 4%).

Also reported: accuracy separately on held-out words the model got right or wrong at the planted slot; a label-shuffle
control (same probe on randomly permuted labels).

Usage: python probe_layers.py N_WORDS     (I used 4000; about 6 minutes in total)
"""
from paths import MODEL_TORN
import csv, sys, time, json
import numpy as np
from np_transformer import NPTransformer
from permdata import plant_batch, apply_words, pointer_path

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
m = NPTransformer(MODEL_TORN)
rng = np.random.default_rng(31337)
n = 25
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def query_hidden(words, ks, bs=100):
    """Hidden states (B, 6, D) at the query position of slot k, and teacher-forced correctness at slot k."""
    perms = apply_words(words)
    B = len(words)
    H = np.zeros((B, m.n_blocks + 1, m.D), np.float32)
    ok = np.zeros(B, bool)
    idx = np.arange(B)
    for a in range(0, B, bs):
        w, p, k = words[a:a + bs], perms[a:a + bs], ks[a:a + bs]
        toks = np.concatenate([w, np.full((len(w), 1), m.START, w.dtype), p[:, :-1] + m.num_trans], axis=1)
        logits, hid = m.forward(toks, return_hidden=True)
        ar = np.arange(len(w))
        for l in range(m.n_blocks + 1):
            H[a:a + len(w), l] = hid[l][ar, 25 + k]
        pred = logits[ar, k].argmax(-1) - m.num_trans
        ok[a:a + len(w)] = pred == p[ar, k]
    return H, ok


def fit_eval(Htr, Ytr, Hva, Yva, Hte, lams=LAMS):
    """Ridge probes for all labels at once (Y: (n, R) integer labels). Returns test predictions (n_te, R)."""
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Xtr, Xva, Xte = (Htr - mu) / sd, (Hva - mu) / sd, (Hte - mu) / sd
    lam, Q = np.linalg.eigh(Xtr.T @ Xtr)
    out = np.zeros((len(Hte), Ytr.shape[1]), np.int64)
    best_l = np.zeros(Ytr.shape[1])
    for r in range(Ytr.shape[1]):
        Y1 = np.eye(n)[Ytr[:, r]]
        XtY = Q.T @ (Xtr.T @ Y1)
        best = (-1, None)
        for lm in lams:
            Wm = Q @ (XtY / (lam + lm)[:, None])
            acc = (np.argmax(Xva @ Wm, 1) == Yva[:, r]).mean()
            if acc > best[0]:
                best = (acc, Wm, lm)
        out[:, r] = np.argmax(Xte @ best[1], 1)
        best_l[r] = best[2]
    return out


designs = [(8, 9), (12, 13), (16, 17), (20, 21), (20, 6)]     # (hops L, distinct points s)
rows, extra = [], []
t0 = time.time()
for L, s in designs:
    words, ks = plant_batch(rng, NW, L, s)
    H, ok = query_hidden(words, ks)
    Y = np.array([pointer_path(w, int(k)) for w, k in zip(words, ks)])        # (NW, L+1)
    assert Y.shape[1] == L + 1 and (Y[:, 0] == ks).all()
    ntr, nva = int(.6 * NW), int(.2 * NW)
    sl = (slice(0, ntr), slice(ntr, ntr + nva), slice(ntr + nva, NW))
    ok_te = ok[sl[2]]
    print(f"[{time.time()-t0:4.0f}s] L={L} s={s}: model correct at planted slot {ok.mean():.3f}", flush=True)
    for l in range(m.n_blocks + 1):
        P = fit_eval(H[sl[0], l], Y[sl[0]], H[sl[1], l], Y[sl[1]], H[sl[2], l])
        Yte = Y[sl[2]]
        for r in range(L + 1):
            hit = P[:, r] == Yte[:, r]
            rows.append([L, s, l, r, int(hit.sum()), len(hit),
                         float(hit[ok_te].mean()) if ok_te.any() else np.nan, int(ok_te.sum()),
                         float(hit[~ok_te].mean()) if (~ok_te).any() else np.nan, int((~ok_te).sum())])
        if l in (0, 5):                                           # label-shuffle control at the extremes
            Ysh = Y[sl[0]][rng.permutation(ntr)]
            Psh = fit_eval(H[sl[0], l], Ysh, H[sl[1], l], Y[sl[1]], H[sl[2], l])
            extra.append(dict(L=L, s=s, layer=l, shuffled_label_acc=float((Psh == Yte).mean())))
    print(f"[{time.time()-t0:4.0f}s]   probes done", flush=True)

with open("probe_results.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["L", "s", "layer", "r", "n_hit", "n_test", "acc_when_model_correct", "n_correct", "acc_when_model_wrong", "n_wrong"])
    wr.writerows(rows)
json.dump(extra, open("probe_controls.json", "w"), indent=1)
print("DONE", flush=True)
