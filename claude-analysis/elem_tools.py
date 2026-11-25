"""
Tools for the ELEMENTARY (adjacent-transposition) model, elem-partitioned-long-boosted-1.
Token convention (verified against the stored permutations of the real data, see elem_data_stats.py):
    token t >= 1 swaps the entries at positions (t-1, t) of the arrangement; token 0 is the identity.
    Output tokens are num_trans + permutation entry, num_trans = 16; START = 33.
Pointer picture: reading the word right to left, a pointer starting at k moves between t-1 and t whenever it sits on one of them.
"""
import numpy as np

G, N = 16, 120


def apply_words_elem(words):
    B = len(words)
    arr = np.tile(np.arange(G), (B, 1))
    rows = np.arange(B)
    for i in range(words.shape[1]):
        t = words[:, i]
        m = t > 0
        r = rows[m]
        a, b = t[m] - 1, t[m]
        x, y = arr[r, a].copy(), arr[r, b].copy()
        arr[r, a], arr[r, b] = y, x
    return arr


def trajectories_elem(words, check=False, hopmask=False):
    """Pointer chase of every output slot. Returns hops (B,G), width (B,G) = number of distinct positions visited."""
    B, T = words.shape
    pos = np.tile(np.arange(G), (B, 1))
    lo, hi = pos.copy(), pos.copy()
    hops = np.zeros((B, G), np.int32)
    hm = np.zeros((B, G, T), bool) if hopmask else None
    for i in range(T - 1, -1, -1):
        tok = words[:, i][:, None]
        a, b = tok - 1, tok
        nz = tok > 0
        to_b, to_a = (pos == a) & nz, (pos == b) & nz
        moved = to_a | to_b
        pos = np.where(to_b, b, np.where(to_a, a, pos))
        hops += moved
        lo = np.minimum(lo, pos); hi = np.maximum(hi, pos)
        if hopmask:
            hm[:, :, i] = moved
    if check:
        assert np.array_equal(pos, apply_words_elem(words)), "pointer picture disagrees with permutation"
    out = dict(hops=hops, width=hi - lo + 1, final=pos)
    if hopmask:
        out["hopmask"] = hm
    return out


def tf_eval_elem(model, words, perms=None, bs=50):
    """Teacher-forced evaluation. Returns perms, correct (B,G), margin (B,G)."""
    if perms is None:
        perms = apply_words_elem(words)
    B = len(words)
    correct = np.zeros((B, G), bool); margin = np.zeros((B, G), np.float32)
    for s in range(0, B, bs):
        p, lg = model.teacher_forced_predictions(words[s:s + bs], perms[s:s + bs])
        tgt = perms[s:s + bs] + model.num_trans
        tl = np.take_along_axis(lg, tgt[..., None], -1)[..., 0]
        l2 = lg.copy(); np.put_along_axis(l2, tgt[..., None], -np.inf, -1)
        margin[s:s + bs] = tl - l2.max(-1)
        correct[s:s + bs] = p == perms[s:s + bs]
    return perms, correct, margin


def touching_tokens(x):
    """Tokens that would move a pointer sitting at position x."""
    return {t for t in (x, x + 1) if 1 <= t <= G - 1}


def plant_elem_word(rng, k, L, w, hop_reads=None):
    """A word in which the pointer of slot k makes exactly L hops and visits exactly w distinct positions (an interval of
    width w). The pointer first runs to the nearer end of the interval, then to the other end, then wanders inside it.
    All non-hop letters avoid the pointer's current position. Requires L >= (w-1) + min distance from k to an end."""
    for _ in range(200):
        lo = int(rng.integers(max(0, k - w + 1), min(k, G - w) + 1))
        hi = lo + w - 1
        e1, e2 = (lo, hi) if (k - lo) <= (hi - k) else (hi, lo)
        need = abs(k - e1) + (w - 1)
        if need <= L:
            break
    else:
        raise ValueError(f"cannot plant L={L}, w={w} for k={k}")
    path = [k]
    x = k
    while x != e1:
        x += 1 if e1 > x else -1; path.append(x)
    while x != e2:
        x += 1 if e2 > x else -1; path.append(x)
    while len(path) - 1 < L:
        opts = [y for y in (x - 1, x + 1) if lo <= y <= hi]
        x = int(rng.choice(opts)); path.append(x)
    assert len(path) - 1 == L
    if hop_reads is None:
        hop_reads = np.sort(rng.choice(N, size=L, replace=False))
    hop_set = {int(r): j for j, r in enumerate(hop_reads)}
    word = np.zeros(N, np.int64)
    x = k
    for r in range(N):
        t = N - 1 - r
        if r in hop_set:
            y = path[hop_set[r] + 1]
            word[t] = max(x, y)
            x = y
        else:
            bad = touching_tokens(x)
            word[t] = int(rng.choice([c for c in range(G) if c not in bad]))
    return word


def plant_elem_batch(rng, B, L, w):
    words, ks = [], []
    while len(words) < B:
        k = int(rng.integers(0, G))
        try:
            words.append(plant_elem_word(rng, k, L, w)); ks.append(k)
        except ValueError:          # this start point cannot reach both ends of a width-w interval in L hops
            continue
    words, ks = np.stack(words), np.array(ks)
    tr = trajectories_elem(words, check=True)
    idx = np.arange(B)
    assert (tr["hops"][idx, ks] == L).all() and (tr["width"][idx, ks] == w).all(), "planting failed"
    return words, ks
