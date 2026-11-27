"""
Why did the naive window method not generalise, when partitioned windows did?

Models from the Spartan folder (configs from the git history of Midataur/permutations):
  elem-big-1, elem-big-2   naive window: an S_10 word of 20 elementary letters, all shifted by one random offset into
                           S_25 (tokens shift..shift+9); test words: 20 letters uniform over the 25 tokens.
  elem-wacky-1, -2         two windows of 5 tokens, each letter in a randomly chosen one of the two (S_25, 20 letters).
  elem-partitioned-1       partitioned windows (S_16, m = 9, 16 letters); test words uniform over the 16 tokens.
All are the legacy architecture (extra attention layer before 4 blocks).

Token t >= 1 swaps positions (t-1, t); token 0 is the identity.

Tests:
 1. Accuracy on the training distribution (validation file) and on the test file, by slot and by word, and test-slot
    accuracy by hop count of the slot's chain.
 2. Far letters. Take a window word (from the validation file) and insert j letters that cannot touch any point the
    window can reach (tokens whose two points are at least 2 away from the window's points), at random places; the
    answers at the window's points are unchanged in meaning. Accuracy at the window's moved slots, j = 0, 1, 2, 4, 8.
    (For j > 0 the word is shortened by j window letters at random to keep its length.)

Usage: python window_failure.py SNAPDIR DATADIR -> window_failure.json
"""
import sys, json
import numpy as np
import np_transformer
from np_transformer import NPTransformer

_orig = np_transformer.load_safetensors


def _load(p):
    """The January/February 2025 models were trained with no attention mask at all (transformer.py at 4ee9609 and
    058a27f): store an all-ones mask so that NPTransformer uses full attention."""
    W, meta = _orig(p)
    W = {k.replace("_orig_mod.", ""): v for k, v in W.items()}
    T = W["position_embedding.weight"].shape[0]
    W.setdefault("blocks.0.sa.heads.0.tril", np.ones((T, T), np.float32))
    return W, meta


np_transformer.load_safetensors = _load

SNAP, DATA = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)


def perms_of(words, G):
    B = len(words); arr = np.tile(np.arange(G), (B, 1)); rows = np.arange(B)
    for i in range(words.shape[1]):
        t = words[:, i]; m = t > 0; r = rows[m]; a, b = t[m] - 1, t[m]
        x, y = arr[r, a].copy(), arr[r, b].copy(); arr[r, a], arr[r, b] = y, x
    return arr


def hops_of(words, G):
    B, T = words.shape; pos = np.tile(np.arange(G), (B, 1)); hops = np.zeros((B, G), int)
    for i in range(T - 1, -1, -1):
        tok = words[:, i][:, None]; nz = tok > 0
        tb, ta = (pos == tok - 1) & nz, (pos == tok) & nz
        pos = np.where(tb, tok, np.where(ta, tok - 1, pos)); hops += tb | ta
    return hops


def evaluate(m, words, G, bs=100):
    """Teacher forcing as the old models were trained (dataloading.py at 4ee9609): input = word, START, the true
    permutation tokens so far, then NULL padding to the full context; the prediction is read at the LAST position."""
    P = perms_of(words, G); C = np.zeros(P.shape, bool); T = words.shape[1]
    NULL, START = m.num_normal, m.num_normal + 1
    for s in range(0, len(words), bs):
        w, pp = words[s:s + bs], P[s:s + bs]; B = len(w)
        toks = np.full((B, G, T + 1 + G), NULL, np.int64)
        toks[:, :, :T] = w[:, None, :]; toks[:, :, T] = START
        for i in range(G):
            toks[:, i, T + 1:T + 1 + i] = pp[:, :i] + m.num_trans
        lg = m.forward(toks.reshape(B * G, -1))[:, -1, :].reshape(B, G, -1)
        C[s:s + bs] = (lg.argmax(-1) - m.num_trans) == pp
    return P, C


def load(name, G, T):
    return NPTransformer(f"{SNAP}/{name}/model.safetensors", max_group=G, input_len=T, n_blocks=4, num_trans=G)


def far_insert(words, G, j):
    """Insert j letters far from each word's window (window = range of its nonzero tokens)."""
    out, keep = [], []
    for w in words:
        nz = w[w > 0]
        lo_pt, hi_pt = nz.min() - 1, nz.max()                     # points the window can move
        far = [t for t in range(1, G) if t - 1 >= hi_pt + 2 or t <= lo_pt - 2]
        if j > 0 and not far:
            continue
        w2 = list(w)
        if j > 0:
            drop = set(rng.choice(len(w2), size=j, replace=False)); w2 = [x for i, x in enumerate(w2) if i not in drop]
            for _ in range(j):
                w2.insert(int(rng.integers(0, len(w2) + 1)), int(rng.choice(far)))
        out.append(w2); keep.append((lo_pt, hi_pt))
    return np.array(out, dtype=np.int64), keep


cfg = {"elem-big-1": ("elem_big", 25, 20), "elem-big-2": ("elem_big", 25, 20),
       "elem-wacky-1": ("elem_wacky", 25, 20), "elem-wacky-2": ("elem_wacky", 25, 20),
       "elem-partitioned-1": ("elem_partitioned", 16, 16)}
res = {}
for name, (folder, G, T) in cfg.items():
    m = load(name, G, T); r = {}
    val = np.loadtxt(f"{DATA}/{folder}/val_data.csv", delimiter=",", dtype=np.int64)[:1000]
    test = np.loadtxt(f"{DATA}/{folder}/test_data.csv", delimiter=",", dtype=np.int64)[:1000]
    for tag, W in (("train distribution (val file)", val), ("test file", test)):
        P, C = evaluate(m, W, G); H = hops_of(W, G)
        r[tag] = dict(words=len(W), slot_acc=float(C.mean()), word_acc=float(C.all(1).mean()),
                      moved_slot_acc=float(C[H > 0].mean()),
                      by_hops={int(h): [float(C[H == h].mean()), int((H == h).sum())] for h in range(0, 8) if (H == h).sum() >= 30})
    # far letters inserted into window words (only meaningful for the single-window data; also run for the others)
    far = {}
    for j in (0, 1, 2, 4, 8):
        W, win = far_insert(val[:600], G, j)
        if len(W) == 0:
            continue
        P, C = evaluate(m, W, G); H = hops_of(W, G)
        inwin = np.zeros_like(C)
        for b, (lo, hi) in enumerate(win):
            inwin[b, max(lo, 0):hi + 1] = True
        far[j] = dict(words=len(W), window_slot_acc=float(C[inwin].mean()), window_moved_slot_acc=float(C[inwin & (H > 0)].mean()),
                      far_slot_acc=float(C[~inwin].mean()), word_acc=float(C.all(1).mean()),
                      window_slot_mean_hops=float(H[inwin].mean()))
    r["far letters"] = far
    res[name] = r
    print(name)
    for tag in ("train distribution (val file)", "test file"):
        d = r[tag]; print(f"  {tag:32s} slots {d['slot_acc']:.4f}  words {d['word_acc']:.3f}  moved slots {d['moved_slot_acc']:.4f}  by hops "
                          + " ".join(f"{h}:{v[0]:.3f}" for h, v in d["by_hops"].items()))
    for j, d in far.items():
        print(f"  far letters j={j}: window slots {d['window_slot_acc']:.4f} (moved {d['window_moved_slot_acc']:.4f}, mean hops {d['window_slot_mean_hops']:.2f})"
              f"  far slots {d['far_slot_acc']:.4f}  words {d['word_acc']:.3f}  ({d['words']} words)")
    sys.stdout.flush()
json.dump(res, open("window_failure.json", "w"), indent=1)
