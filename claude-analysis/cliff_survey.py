"""
The hop-count cliff and the memorisation gap across the general-transposition models in the Spartan model folder.

Models: torn-bigger-5 ... -12 (snapshots; the older legacy architecture with an extra attention layer before the
blocks, 4 or 8 blocks, learning rate 3e-5, trained on 14 or 18 files of the torn_bigger data), torn-bigger-13 as in
the repo (the February 2025 checkpoint used in the report) and torn-bigger-13 as it is now on Spartan (after the
24-epoch resume in November 2025).  torn-bigger-1 to -4 are unmasked older models and are left out.

For each model:
  1. planted chains (section 3.3): L = 8 ... 24 hops through s = min(L+1, 10) points, teacher-forced accuracy at the
     planted slot;
  2. fresh natural slots with h = 11 ... 19 hops (natural 10-point words relabelled into 25, collected by rejection,
     the same words for every model);
  3. the model's own training slots with h = 11 ... 18 hops (Test 16, section 3.13 of the report).

Usage: python cliff_survey.py SNAPSHOT_DIR DATA14_18_DIR DATA20_DIR [N_PLANT] [CAP]
  DATA14_18_DIR: train_data1..18.csv of commit 2c8e819 (files 1-14 are the set of torn-bigger-5 and -6)
  DATA20_DIR:    train_data1..20.csv of commit 5863de5 (torn-bigger-13)
Writes cliff_survey.json and cliff_survey_log.txt.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from np_transformer import NPTransformer
from permdata import plant_batch
from evalutil import tf_eval

n, N = 25, 25
snap, d18, d20 = sys.argv[1], sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 200
CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 300
LOG = open("cliff_survey_log.txt", "a")
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:6.0f}s] {s}", flush=True); LOG.write(s + "\n"); LOG.flush()


def hops_of(words):
    B = len(words)
    pos = np.tile(np.arange(n, dtype=np.int16), (B, 1)); hops = np.zeros((B, n), np.int8)
    for t in range(N - 1, -1, -1):
        a = (words[:, t] // n)[:, None]; b = (words[:, t] % n)[:, None]; nz = a != b
        to_b = (pos == a) & nz; to_a = (pos == b) & nz
        hops += (to_a | to_b); pos = np.where(to_b, b, np.where(to_a, a, pos)).astype(np.int16)
    return hops


def natural10(rng, B):
    a = rng.integers(0, 10, (B, N)); b = rng.integers(0, 10, (B, N)); rel = rng.random((B, n)).argsort(1)
    return (np.take_along_axis(rel, a, 1) * n + np.take_along_axis(rel, b, 1)).astype(np.int16)


def select(words, H, cap, rng, hmin=11):
    """(words, slots, h) with at most cap slots per h >= hmin."""
    wi, si = np.nonzero(H >= hmin); hh = H[wi, si]; keep = []
    for h in np.unique(hh):
        idx = rng.permutation(np.nonzero(hh == h)[0])[:cap]; keep += list(idx)
    keep = np.array(sorted(keep))
    return words[wi[keep]].astype(np.int64), si[keep], hh[keep].astype(int)


# ---- fresh natural slots (cached)
fresh_f = "/home/claude/snapwork/cliff_fresh.npz"
if os.path.exists(fresh_f):
    z = np.load(fresh_f); FW, FS, FH = z["w"], z["s"], z["h"]
else:
    rng = np.random.default_rng(777); Ws, Ss, Hs = [], [], []; cnt = {}
    for it in range(40):
        W = natural10(rng, 1_000_000); H = hops_of(W); wi, si = np.nonzero(H >= 11)
        for w, s in zip(wi, si):
            h = int(H[w, s])
            if cnt.get(h, 0) < CAP:
                cnt[h] = cnt.get(h, 0) + 1; Ws.append(W[w]); Ss.append(s); Hs.append(h)
    FW, FS, FH = np.array(Ws, np.int64), np.array(Ss), np.array(Hs)
    np.savez(fresh_f, w=FW, s=FS, h=FH)
log("fresh slots by h:", {int(h): int((FH == h).sum()) for h in np.unique(FH)})


def train_slots(d, nfiles, tag):
    f = f"/home/claude/snapwork/cliff_train_{tag}.npz"
    if os.path.exists(f):
        z = np.load(f); return z["w"], z["s"], z["h"]
    W = np.concatenate([pd.read_csv(f"{d}/train_data{i}.csv", header=None, dtype=np.int16).values for i in range(1, nfiles + 1)])
    H = np.concatenate([hops_of(W[s:s + 500_000]) for s in range(0, len(W), 500_000)])
    w, s, h = select(W, H, CAP, np.random.default_rng(5))
    np.savez(f, w=w, s=s, h=h)
    log(f"training set {tag}: {len(W):,} words; slots with h>=11..18:", {k: int((H >= k).sum()) for k in range(11, 19)})
    return w, s, h


inv = {d["name"]: d for d in json.load(open("/home/claude/snapwork/inventory.json"))}
models = [(f"torn-bigger-{i}", os.path.join(snap, f"torn-bigger-{i}", "model.safetensors"), inv[f"torn-bigger-{i}"]["blocks"],
           (d18, 14 if i in (5, 6) else 18, "14" if i in (5, 6) else "18")) for i in range(5, 13)]
models += [("torn-bigger-13 (repo, Feb 2025)", "../scaling-generator/trained-models/torn-bigger-13/model.safetensors", 5, (d20, 20, "20")),
           ("torn-bigger-13 (Spartan, after resume)", os.path.join(snap, "torn-bigger-13", "model.safetensors"), 5, (d20, 20, "20"))]
out = json.load(open("cliff_survey.json")) if os.path.exists("cliff_survey.json") else {}


def acc_by_h(m, W, S, Hh):
    _, corr, _, _ = tf_eval(m, W, bs=200)
    c = corr[np.arange(len(W)), S]
    return {int(h): [int(c[Hh == h].sum()), int((Hh == h).sum())] for h in np.unique(Hh)}


for name, path, blocks, (dd, nf, tag) in models:
    if name in out:
        continue
    m = NPTransformer(path, n_blocks=blocks)
    r = dict(blocks=blocks, legacy=m.legacy is not None, train_files=nf, planted={})
    rng = np.random.default_rng(4242)
    for L in (8, 12, 14, 16, 18, 20, 24):
        w, k = plant_batch(rng, NP, L, min(L + 1, 10))
        _, corr, _, _ = tf_eval(m, w, bs=200)
        r["planted"][L] = float(corr[np.arange(NP), k].mean())
    r["fresh"] = acc_by_h(m, FW, FS, FH)
    TW, TS, TH = train_slots(dd, nf, tag)
    r["train"] = acc_by_h(m, TW, TS, TH)
    out[name] = r
    json.dump(out, open("cliff_survey.json", "w"), indent=1)
    pl = "  ".join(f"L{L}:{v:.2f}" for L, v in r["planted"].items())
    fr = "  ".join(f"h{h}:{a / b:.3f}" for h, (a, b) in r["fresh"].items() if h <= 17)
    tr = "  ".join(f"h{h}:{a / b:.3f}" for h, (a, b) in r["train"].items() if h <= 17)
    log(f"{name} ({blocks} blocks{', legacy' if r['legacy'] else ''}, {nf} files)\n   planted  {pl}\n   fresh    {fr}\n   training {tr}")
log("DONE")
