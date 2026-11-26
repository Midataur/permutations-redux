"""
Test 16: does torn-bigger-13 know its own long chains better than fresh ones of the same length?

The original model was trained for 185 epochs on a static file of 8M natural words (20 files of 400,000 words,
commit 5863de5 of Midataur/permutations, scaling-generator/data/torn_bigger/), and its wandb history shows its
training loss falling to ~4e-9 while the held-out loss stays ~1e-5 (memorisation of the file).  Here:

  1. every output slot of all 8M training words gets its pointer-chase hop count h (as in permdata.trajectories);
  2. teacher-forced accuracy and margin at the slot are measured for training slots with h >= 12, and for slots with
     the same h in held-out words: (a) the run's own validation file (100,000 words, same generator, never trained
     on), (b) fresh natural words over 10 points relabelled into 25 (the training distribution), collected by
     rejection until each h has enough slots;
  3. as a baseline, margins on random training words and random fresh words at every h.

If the model is right on its training slots at h = 14-17 and much less on fresh slots of the same h, it has memorised
the long chains of its file rather than learned to follow them.

Usage: python run_memorisation.py DATA_DIR [N_FRESH_MAX_WORDS]
  DATA_DIR holds train_data1.csv ... train_data20.csv and val_data.csv
Writes memorisation_results.json and memorisation_log.txt.  With TORN_MODEL=<checkpoint> and MEMO_TAG=<tag> the same
words (same random seed) are run through another checkpoint, e.g. the fresh-data control g_m10, for which the
"training" slots are simply more held-out words; outputs then get the suffix _<tag>.
"""
import sys, json, time
import numpy as np
import pandas as pd
from paths import MODEL_TORN
from np_transformer import NPTransformer
from evalutil import tf_eval

n, N = 25, 25
D = sys.argv[1]
FRESH_MAX = int(float(sys.argv[2])) if len(sys.argv) > 2 else 60_000_000
CAP = 1500            # slots per hop count and source evaluated with the model
H0 = 12
rng = np.random.default_rng(20261127)
import os
TAG = os.environ.get("MEMO_TAG", "")
SUF = f"_{TAG}" if TAG else ""
LOG = open(f"memorisation_log{SUF}.txt", "w")
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)
    LOG.write(s + "\n"); LOG.flush()


def hops_of(words):
    """Hop count of every output slot (B, n), right-to-left chase as in permdata.trajectories."""
    B = len(words)
    pos = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    hops = np.zeros((B, n), np.int8)
    for t in range(N - 1, -1, -1):
        a = (words[:, t] // n)[:, None]; b = (words[:, t] % n)[:, None]
        nz = a != b
        to_b = (pos == a) & nz; to_a = (pos == b) & nz
        hops += (to_a | to_b)
        pos = np.where(to_b, b, np.where(to_a, a, pos)).astype(np.int16)
    return hops


def load(f):
    return pd.read_csv(f"{D}/{f}", header=None, dtype=np.int16).values


def natural10(B):
    a = rng.integers(0, 10, (B, N)); b = rng.integers(0, 10, (B, N))
    rel = rng.random((B, n)).argsort(1)
    return (np.take_along_axis(rel, a, 1) * n + np.take_along_axis(rel, b, 1)).astype(np.int16)


# ---------------------------------------------------------------- 1. training words
train = np.concatenate([load(f"train_data{i}.csv") for i in range(1, 21)])
val = load("val_data.csv")
log(f"training words {len(train):,}, validation words {len(val):,}")
pts = lambda W: np.array([len(np.unique(np.concatenate([w // n, w % n]))) for w in W[:20000]])
log("points touched (first 20k words): train max", pts(train).max(), " val max", pts(val).max())
# duplicates
v = lambda W: np.ascontiguousarray(W).view(np.dtype((np.void, W.dtype.itemsize * W.shape[1]))).ravel()
tv = v(train); uq = np.unique(tv)
log(f"distinct training words {len(uq):,} of {len(train):,}; validation words also in training: "
    f"{np.isin(v(val), uq).sum()}")

Htr = np.concatenate([hops_of(train[s:s + 500_000]) for s in range(0, len(train), 500_000)])
Hva = hops_of(val)
log("training slots by hops >= h:", {h: int((Htr >= h).sum()) for h in range(10, 20)})
log("validation slots by hops >= h:", {h: int((Hva >= h).sum()) for h in range(10, 20)})


def pick(H, cap, hmin=H0):
    """(word, slot, h) for slots with h >= hmin, at most cap per h, random order."""
    wi, si = np.nonzero(H >= hmin)
    hh = H[wi, si]
    out = []
    for h in np.unique(hh):
        idx = np.nonzero(hh == h)[0]
        idx = rng.permutation(idx)[:cap]
        out += [(int(wi[i]), int(si[i]), int(h)) for i in idx]
    return out


sel_tr = pick(Htr, CAP)
sel_va = pick(Hva, CAP)

# ---------------------------------------------------------------- 2. fresh words by rejection
fresh_words, fresh_sel = [], []
cnt = {}
made = 0
while made < FRESH_MAX:
    W = natural10(1_000_000); made += len(W)
    H = hops_of(W)
    wi, si = np.nonzero(H >= H0)
    for w, s in zip(wi, si):
        h = int(H[w, s])
        if cnt.get(h, 0) >= CAP:
            continue
        cnt[h] = cnt.get(h, 0) + 1
        fresh_sel.append((len(fresh_words), int(s), h)); fresh_words.append(W[w])
    if made % 10_000_000 == 0:
        log(f"fresh words {made:,}: slots kept by h {dict(sorted(cnt.items()))}")
    if all(cnt.get(h, 0) >= CAP for h in range(H0, 16)) and cnt.get(16, 0) >= 300:
        break
fresh_words = np.array(fresh_words)
log(f"fresh words generated {made:,}; kept {dict(sorted(cnt.items()))}")
# check that the fresh generator matches the training distribution on the tail
Wc = natural10(2_000_000); Hc = hops_of(Wc)
for h in (8, 10, 12, 14):
    log(f"P(h >= {h}) per slot: training {np.mean(Htr >= h):.3e}  fresh {np.mean(Hc >= h):.3e}  "
        f"Binomial(25,0.18) exact see report")

# ---------------------------------------------------------------- 3. the model
model = NPTransformer(MODEL_TORN)


def evaluate(words, sel, name):
    wids = sorted({w for w, _, _ in sel})
    remap = {w: i for i, w in enumerate(wids)}
    _, corr, marg, _ = tf_eval(model, words[wids].astype(np.int64), bs=200)
    rows = [(h, bool(corr[remap[w], s]), float(marg[remap[w], s])) for w, s, h in sel]
    res = {}
    for h in sorted({r[0] for r in rows}):
        c = np.array([r[1] for r in rows if r[0] == h]); m = np.array([r[2] for r in rows if r[0] == h])
        res[h] = dict(n=int(len(c)), acc=float(c.mean()), margin_median=float(np.median(m)),
                      margin_p10=float(np.percentile(m, 10)))
        log(f"{name:10s} h={h:2d}  n={len(c):5d}  accuracy {c.mean():.3f}  median margin {np.median(m):6.2f}  "
            f"10th pct {np.percentile(m, 10):6.2f}")
    return res


R = dict(n_train_words=int(len(train)), n_val_words=int(len(val)), fresh_generated=int(made))
R["train"] = evaluate(train, sel_tr, "training")
R["val"] = evaluate(val, sel_va, "validation")
R["fresh"] = evaluate(fresh_words, fresh_sel, "fresh")

# baseline over all h: random training words against random fresh words
idx = rng.choice(len(train), 4000, replace=False)
Wf = natural10(4000)
base = {}
for name, W, H in (("train", train[idx], Htr[idx]), ("fresh", Wf, hops_of(Wf))):
    _, corr, marg, _ = tf_eval(model, W.astype(np.int64), bs=200)
    base[name] = {}
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)):
        m = (H >= lo) & (H < hi)
        if m.sum():
            base[name][f"{lo}-{hi - 1}"] = dict(n=int(m.sum()), acc=float(corr[m].mean()),
                                               margin_median=float(np.median(marg[m])))
            log(f"baseline {name:6s} h={lo}-{hi - 1}: n={m.sum():6d} acc {corr[m].mean():.5f} "
                f"median margin {np.median(marg[m]):6.2f}")
R["baseline"] = base
R["model"] = MODEL_TORN
json.dump(R, open(f"memorisation_results{SUF}.json", "w"), indent=1)
log("DONE")
