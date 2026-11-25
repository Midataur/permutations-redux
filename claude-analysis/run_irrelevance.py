"""
Test 2: irrelevance interventions on torn-bigger-13 (general transpositions).

For output slot k of a word, a letter is IRRELEVANT if, at its turn in the right-to-left pointer chase, it does not
contain the pointer's current position: it cannot change p_k. We replace irrelevant letters by other letters that also
avoid the pointer, so the true p_k is unchanged by construction, and ask whether the model's prediction at slot k changes.
The output prefix fed to the model is always the ORIGINAL word's true outputs (so only the input letters differ).

Conditions per word set:
  full      all irrelevant letters replaced by random pointer-avoiding letters
  half      each irrelevant letter replaced with probability 1/2
  identity  all irrelevant letters replaced by identity tokens (a == b)
  delete1   positive control: ONE hop letter replaced by an identity token (the true p_k then changes)
Word sets: planted chases with L = 8, 12, 16, 20 hops (s = L+1 distinct points) and natural random words (random slot).
Then an influence map: replace ONE letter at a time (planted L = 12 and 20) and record how often the prediction changes,
for hop letters and for irrelevant letters.

Usage: python run_irrelevance.py N_WORDS N_MAP      (I used 800 and 300; about 12 minutes)
"""
from paths import MODEL_TORN
import sys, csv, json, time
import numpy as np
from np_transformer import NPTransformer
from permdata import plant_batch, random_words, apply_words, trajectories

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 800
NMAP = int(sys.argv[2]) if len(sys.argv) > 2 else 300
m = NPTransformer(MODEL_TORN)
rng = np.random.default_rng(777)
n = N = 25


def hop_mask(word, k):
    """Boolean (N,) array: True where the letter moves the pointer of slot k (right-to-left chase), and the
    pointer position just BEFORE reading each letter."""
    x, hop, at = k, np.zeros(N, bool), np.zeros(N, int)
    for t in range(N - 1, -1, -1):
        a, b = word[t] // n, word[t] % n
        at[t] = x
        if a != b and x in (a, b):
            hop[t] = True
            x = b if x == a else a
    return hop, at


def random_avoiding(x):
    others = [p for p in range(n) if p != x]
    return int(rng.choice(others)) * n + int(rng.choice(others))


def intervene(word, k, mode):
    hop, at = hop_mask(word, k)
    new = word.copy()
    irr = np.nonzero(~hop)[0]
    if mode in ("full", "half"):
        for t in irr:
            if mode == "full" or rng.random() < .5:
                new[t] = random_avoiding(at[t])
    elif mode == "identity":
        for t in irr:
            c = int(rng.integers(n)); new[t] = c * n + c
    elif mode == "delete1":
        h = np.nonzero(hop)[0]
        if len(h):
            c = int(rng.integers(n)); new[int(rng.choice(h))] = c * n + c
    return new


def slot_pred(words, perms, ks, bs=100):
    """Teacher-forced prediction and margin at slot k, with the given output prefix."""
    B = len(words)
    pred = np.zeros(B, int); margin = np.zeros(B)
    ar = np.arange(B)
    for a in range(0, B, bs):
        w, p, k = words[a:a + bs], perms[a:a + bs], ks[a:a + bs]
        toks = np.concatenate([w, np.full((len(w), 1), m.START, w.dtype), p[:, :-1] + m.num_trans], axis=1)
        lg = m.forward(toks)[np.arange(len(w)), k]                  # (b, V)
        pr = lg.argmax(-1)
        tgt = p[np.arange(len(w)), k] + m.num_trans
        tl = lg[np.arange(len(w)), tgt]
        l2 = lg.copy(); l2[np.arange(len(w)), tgt] = -np.inf
        pred[a:a + len(w)] = pr - m.num_trans; margin[a:a + len(w)] = tl - l2.max(-1)
    return pred, margin


t0 = time.time()
sets = []
for L in (8, 12, 16, 20):
    w, k = plant_batch(rng, NW, L, L + 1)
    sets.append((f"planted L={L}", w, k))
wn = random_words(rng, NW); kn = rng.integers(0, n, size=NW)
sets.append(("natural", wn, kn))

rows = []
for name, W, K in sets:
    P = apply_words(W)
    true_k = P[np.arange(len(W)), K]
    p0, m0 = slot_pred(W, P, K)
    nirr = np.array([(~hop_mask(w, int(k))[0]).sum() for w, k in zip(W, K)])
    hops = np.array([hop_mask(w, int(k))[0].sum() for w, k in zip(W, K)])
    print(f"[{time.time()-t0:4.0f}s] {name}: baseline accuracy {np.mean(p0 == true_k):.3f}; mean hops {hops.mean():.1f}, mean irrelevant letters {nirr.mean():.1f}", flush=True)
    for mode in ("full", "half", "identity", "delete1"):
        W2 = np.stack([intervene(w, int(k), mode) for w, k in zip(W, K)])
        if mode != "delete1":
            assert (apply_words(W2)[np.arange(len(W)), K] == true_k).all(), "intervention changed the true p_k"
        p1, m1 = slot_pred(W2, P, K)                                # SAME (original) prefix
        agree = float(np.mean(p1 == p0))
        rows.append(dict(set=name, mode=mode, n=len(W), baseline_acc=float(np.mean(p0 == true_k)),
                         acc_after=float(np.mean(p1 == true_k)), agree=agree,
                         changed=int(np.sum(p1 != p0)),
                         agree_confident=float(np.mean((p1 == p0)[m0 > 5])) if (m0 > 5).any() else None, n_confident=int((m0 > 5).sum()),
                         agree_unconfident=float(np.mean((p1 == p0)[m0 <= 5])) if (m0 <= 5).any() else None, n_unconfident=int((m0 <= 5).sum()),
                         margin_before=float(np.median(m0)), margin_after=float(np.median(m1)),
                         mean_irrelevant=float(nirr.mean())))
        rows[-1]["note"] = "confident = baseline logit margin > 5"
        print(f"   {mode:8s}: prediction unchanged in {100*agree:5.1f}%  accuracy {100*np.mean(p0==true_k):.1f}% -> {100*np.mean(p1==true_k):.1f}%  "
              f"median margin {np.median(m0):.1f} -> {np.median(m1):.1f}", flush=True)
json.dump(rows, open("irrelevance_results.json", "w"), indent=1)

# ---- influence map: one letter at a time
maps = {}
for L in (12, 20):
    W, K = plant_batch(rng, NMAP, L, L + 1)
    P = apply_words(W)
    p0, _ = slot_pred(W, P, K)
    hop_flags = np.stack([hop_mask(w, int(k))[0] for w, k in zip(W, K)])
    ats = np.stack([hop_mask(w, int(k))[1] for w, k in zip(W, K)])
    changed = np.zeros((NMAP, N), bool)
    for t in range(N):
        W2 = W.copy()
        for i in range(NMAP):
            if hop_flags[i, t]:
                c = int(rng.integers(n)); W2[i, t] = c * n + c                 # delete a hop
            else:
                W2[i, t] = random_avoiding(int(ats[i, t]))                      # harmless resample
        p1, _ = slot_pred(W2, P, K)
        changed[:, t] = p1 != p0
    maps[L] = dict(hop_flip=float(changed[hop_flags].mean()), irr_flip=float(changed[~hop_flags].mean()),
                   n_hop=int(hop_flags.sum()), n_irr=int((~hop_flags).sum()),
                   by_position_hop=[float(changed[:, t][hop_flags[:, t]].mean()) if hop_flags[:, t].any() else None for t in range(N)],
                   by_position_irr=[float(changed[:, t][~hop_flags[:, t]].mean()) if (~hop_flags[:, t]).any() else None for t in range(N)])
    print(f"[{time.time()-t0:4.0f}s] influence map L={L}: replacing a HOP letter changes the prediction in {100*maps[L]['hop_flip']:.1f}% of cases; "
          f"replacing an IRRELEVANT letter in {100*maps[L]['irr_flip']:.1f}%", flush=True)
json.dump(maps, open("influence_map.json", "w"), indent=1)
print("DONE", flush=True)
