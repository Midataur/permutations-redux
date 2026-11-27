"""
Beyond (hops, points): does the order of revisits or the placement of the hop letters matter?

A. Walk shape at fixed L hops and s points, hop letters spread at random over the word:
     cycle      a b c d e a b c d e ...         (never returns to the point it just left)
     ping-pong  a b a b ... with the other points visited once each, spread along the walk
     random, split by the share of hops that return to the point just left (reversals): lowest and highest third
B. Placement of the hop letters, at L = 16, s = 10, random walk: spread at random; one contiguous block at the end of
   the word (read first), at the start (read last) or in the middle; every other letter.
C. Fresh natural slots (cached set of cliff_survey.py, 11-19 hops): accuracy by reversal share and by where the hop
   letters sit, within each hop count.

Usage: python pattern_families.py N_PER_CELL name=path:blocks ...  -> pattern_families.json
"""
import sys, json
import numpy as np
from np_transformer import NPTransformer
from permdata import trajectories
from evalutil import tf_eval

NC = int(sys.argv[1]); specs = [a.split("=", 1) for a in sys.argv[2:]]
n = N = 25


def random_walk(rng, L, s):
    new = np.zeros(L, bool); new[0] = True
    if s > 2:
        new[rng.choice(np.arange(1, L), size=s - 2, replace=False)] = True
    pts = rng.permutation(n)[:s]; walk = [pts[0]]; visited = [pts[0]]; nxt = 1
    for h in range(L):
        if new[h]:
            y = pts[nxt]; nxt += 1; visited.append(y)
        else:
            y = rng.choice([p for p in visited if p != walk[-1]])
        walk.append(y)
    return walk


def cycle_walk(rng, L, s):
    pts = rng.permutation(n)[:s]
    return [pts[i % s] for i in range(L + 1)]


def pingpong_walk(rng, L, s):
    pts = rng.permutation(n)[:s]; a, b = pts[0], pts[1]
    others = list(pts[2:])
    slots = sorted(rng.choice(np.arange(2, L + 1), size=len(others), replace=False)) if others else []
    walk = [a]; prev2 = None
    for i in range(1, L + 1):
        if slots and i == slots[0]:
            slots.pop(0); y = others.pop(0)
        else:
            y = b if walk[-1] != b else a
            if y == walk[-1]:
                y = a
        if y == walk[-1]:
            y = b if walk[-1] == a else a
        walk.append(y)
    return walk


def reversal_share(walk):
    L = len(walk) - 1
    return np.mean([walk[i + 1] == walk[i - 1] for i in range(1, L)]) if L > 1 else 0.0


def word_from_walk(rng, walk, reads):
    """Hop h is the letter read at reading index reads[h] (0 = last letter of the word); other letters avoid the pointer."""
    L = len(walk) - 1; word = np.zeros(N, np.int64); hop = {int(r): j for j, r in enumerate(sorted(reads))}; x = walk[0]
    for r in range(N):
        t = N - 1 - r
        if r in hop:
            y = walk[hop[r] + 1]; a, b = (x, y) if rng.random() < 0.5 else (y, x); word[t] = a * n + b; x = y
        else:
            others = [p for p in range(n) if p != x]; word[t] = rng.choice(others) * n + rng.choice(others)
    return word


def batch(rng, make_walk, L, s, reads_fn, want=None):
    W, K, rs = [], [], []
    while len(W) < NC:
        w = make_walk(rng, L, s)
        if len(set(w)) != s or any(w[i] == w[i + 1] for i in range(L)):
            continue
        r = reversal_share(w)
        if want is not None and not want(r):
            continue
        W.append(word_from_walk(rng, w, reads_fn(rng, L))); K.append(w[0]); rs.append(r)
    W, K = np.stack(W), np.array(K)
    tr = trajectories(W, check=True)
    assert (tr["hops"][np.arange(NC), K] == L).all() and (tr["distinct"][np.arange(NC), K] == s).all()
    return W, K, float(np.mean(rs))


spread = lambda rng, L: rng.choice(N, size=L, replace=False)
block = lambda start: (lambda rng, L: np.arange(start, start + L))      # reading indices: 0 = last letter of the word
every_other = lambda rng, L: np.arange(0, 2 * L, 2)[:L] if 2 * L - 1 < N else np.arange(L)

# reversal-share thirds of random walks, per (L, s)
def thirds(L, s, rng=np.random.default_rng(5)):
    r = [reversal_share(random_walk(rng, L, s)) for _ in range(4000)]
    return np.quantile(r, [1 / 3, 2 / 3])

out = {}
for name, rest in specs:
    path, blocks = rest.rsplit(":", 1); m = NPTransformer(path, n_blocks=int(blocks)); res = {"walk": {}, "placement": {}}
    def acc(W, K):
        _, c, mg, _ = tf_eval(m, W, bs=200); i = np.arange(len(W)); return float(c[i, K].mean()), float(np.median(mg[i, K]))
    for L, s in ((16, 5), (18, 5), (16, 10), (20, 10)):
        q1, q2 = thirds(L, s)
        fam = {"cycle": (cycle_walk, None), "random, fewest reversals": (random_walk, lambda r: r <= q1),
               "random, most reversals": (random_walk, lambda r: r >= q2), "ping-pong": (pingpong_walk, None)}
        for fname, (mk, want) in fam.items():
            rng = np.random.default_rng(hash((L, s, fname)) % 2**32)
            W, K, rmean = batch(rng, mk, L, s, spread, want)
            a, g = acc(W, K); res["walk"][f"L={L}, s={s}: {fname}"] = dict(acc=a, margin=g, reversal_share=rmean)
    for pname, fn in (("spread at random", spread), ("block read first (end of word)", block(0)),
                      ("block in the middle", block(4)), ("block read last (start of word)", block(9)), ("every other letter", None)):
        rng = np.random.default_rng(hash(pname) % 2**32)
        reads = fn if fn is not None else (lambda rng, L: np.arange(0, 25, 2)[:L] if L <= 13 else np.sort(rng.choice(N, size=L, replace=False)))
        if pname == "every other letter":
            Lp, sp = 12, 10
        else:
            Lp, sp = 16, 10
        W, K, _ = batch(rng, random_walk, Lp, sp, reads)
        a, g = acc(W, K); res["placement"][f"L={Lp}, s={sp}: {pname}"] = dict(acc=a, margin=g)
    out[name] = res
    print(name)
    for k, v in res["walk"].items():
        print(f"   {k:42s} acc {v['acc']:.3f}  margin {v['margin']:5.1f}  reversal share {v['reversal_share']:.2f}")
    for k, v in res["placement"].items():
        print(f"   {k:42s} acc {v['acc']:.3f}  margin {v['margin']:5.1f}")
    sys.stdout.flush()

# C. natural fresh slots: what goes with errors of the first model, within hop counts
z = np.load("/home/claude/snapwork/cliff_fresh.npz"); FW, FS, FH = z["w"].astype(np.int64), z["s"], z["h"]
tr = trajectories(FW, check=False); FD = tr["distinct"][np.arange(len(FW)), FS]
hm = tr["hopmask"][np.arange(len(FW)), FS]                      # (B, 25) which letters are hops of this slot
mean_pos = np.array([np.mean(np.nonzero(h)[0]) for h in hm])    # mean word position of the hop letters
rev = []
for w, k in zip(FW, FS):
    pos = k; walk = [pos]
    for t in range(N - 1, -1, -1):
        a, b = w[t] // n, w[t] % n
        if a != b and pos in (a, b):
            pos = b if pos == a else a; walk.append(pos)
    rev.append(reversal_share(walk))
rev = np.array(rev)
natural = {}
for name, rest in specs:
    path, blocks = rest.rsplit(":", 1); m = NPTransformer(path, n_blocks=int(blocks))
    _, c, _, _ = tf_eval(m, FW, bs=200); ok = c[np.arange(len(FW)), FS]
    natural[name] = {}
    for h in (14, 15, 16):
        sel = FH == h
        rq = np.quantile(rev[sel], [0.5]); pq = np.quantile(mean_pos[sel], [0.5])
        natural[name][h] = dict(n=int(sel.sum()),
                                low_reversal=float(ok[sel & (rev <= rq[0])].mean()), high_reversal=float(ok[sel & (rev > rq[0])].mean()),
                                hops_early_in_word=float(ok[sel & (mean_pos <= pq[0])].mean()), hops_late_in_word=float(ok[sel & (mean_pos > pq[0])].mean()))
    print(name, "natural slots:", {h: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for h, d in natural[name].items()})
json.dump(dict(cells=out, natural=natural), open("pattern_families.json", "w"), indent=1)
