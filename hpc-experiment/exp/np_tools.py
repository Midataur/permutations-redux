"""NumPy helpers (CPU): permutations, pointer chases and chain planting, for BOTH problem types.

General transpositions (n=25): token = a*n + b is the transposition of positions a and b (a == b is the identity).
Elementary transpositions (G=16): token t >= 1 swaps positions t-1 and t, token 0 is the identity.
Reading order: the pointer chase reads a word right to left (reading index r = 0 is the LAST letter).
"""
import numpy as np

# ------------------------------------------------------------------------------------------ general
def apply_words(words, n=25):
    B, N = words.shape
    perm = np.tile(np.arange(n), (B, 1))
    rows = np.arange(B)
    for t in range(N):
        a, b = words[:, t] // n, words[:, t] % n
        pa, pb = perm[rows, a].copy(), perm[rows, b].copy()
        perm[rows, a] = pb
        perm[rows, b] = pa
    return perm


def trajectories(words, n=25, check=True):
    """Pointer chase of every output slot k. Returns hops (B,n) and distinct (B,n) = distinct points visited incl. start."""
    B, N = words.shape
    pos = np.tile(np.arange(n), (B, 1))
    hops = np.zeros((B, n), dtype=np.int32)
    visited = np.zeros((B, n, n), dtype=bool)
    visited[:, np.arange(n), np.arange(n)] = True
    for t in range(N - 1, -1, -1):
        a, b = (words[:, t] // n)[:, None], (words[:, t] % n)[:, None]
        nz = a != b
        to_b, to_a = (pos == a) & nz, (pos == b) & nz
        moved = to_a | to_b
        pos = np.where(to_b, b, np.where(to_a, a, pos))
        hops += moved
        bi, ki = np.nonzero(moved)
        visited[bi, ki, pos[bi, ki]] = True
    if check:
        assert np.array_equal(pos, apply_words(words, n)), "pointer picture disagrees with permutation"
    return dict(hops=hops, distinct=visited.sum(2))


def natural_words(rng, B, m, n=25, N=25):
    """Words over m points (uniform over the m^2 ordered pairs, a == b allowed), randomly relabelled into n points.
    m = n means no restriction. This is what the repo's generator does with -g m -M n -S -R."""
    a = rng.integers(0, m, (B, N)); b = rng.integers(0, m, (B, N))
    rel = np.argsort(rng.random((B, n)), 1)
    return np.take_along_axis(rel, a, 1) * n + np.take_along_axis(rel, b, 1)


def plant_word(rng, k, L, s, n=25, N=25, pts=None):
    """One word in which the pointer of output slot k makes exactly L hops and visits exactly s distinct points
    (including its start). Every non-hop letter avoids the pointer's current position, so it cannot move it.
    pts: optional list of allowed points (must contain k); all letters then live on these points only.
    Requires 2 <= s <= min(L+1, len(pts))."""
    pts = list(range(n)) if pts is None else list(pts)
    assert k in pts and 2 <= s <= min(L + 1, len(pts)) and L <= N
    hop_reads = np.sort(rng.choice(N, size=L, replace=False))
    hop_set = set(int(r) for r in hop_reads)
    new_flag = np.zeros(L, dtype=bool)
    new_flag[0] = True
    if s > 2:
        new_flag[rng.choice(np.arange(1, L), size=s - 2, replace=False)] = True
    x, visited, word, h = k, [k], np.zeros(N, dtype=np.int64), 0
    for r in range(N):
        t = N - 1 - r
        if r in hop_set:
            if new_flag[h]:
                y = int(rng.choice([p for p in pts if p not in visited])); visited.append(y)
            else:
                y = int(rng.choice([p for p in visited if p != x]))
            a, b = (x, y) if rng.random() < 0.5 else (y, x)
            word[t] = a * n + b
            x, h = y, h + 1
        else:
            others = [p for p in pts if p != x]
            word[t] = int(rng.choice(others)) * n + int(rng.choice(others))
    return word


def plant_batch(rng, B, L, s, n=25, N=25, support=None):
    """B planted words. support=None: letters over all n points; support=m: each word lives on a random m-subset."""
    words, ks = [], rng.integers(0, n, size=B)
    for k in ks:
        pts = None
        if support is not None:
            others = [p for p in range(n) if p != k]
            pts = [int(k)] + [int(p) for p in rng.choice(others, size=support - 1, replace=False)]
        words.append(plant_word(rng, int(k), L, s, n, N, pts))
    words = np.stack(words)
    tr = trajectories(words, n)
    idx = np.arange(B)
    assert (tr["hops"][idx, ks] == L).all() and (tr["distinct"][idx, ks] == s).all(), "planting failed"
    return words, ks


# ------------------------------------------------------------------------------------------ elementary
G_E, N_E = 16, 120


def apply_words_elem(words, G=G_E):
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


def trajectories_elem(words, G=G_E, check=False):
    """Pointer chase of every output slot. hops (B,G); width (B,G) = number of adjacent positions the chase spans."""
    B, T = words.shape
    pos = np.tile(np.arange(G), (B, 1))
    lo, hi = pos.copy(), pos.copy()
    hops = np.zeros((B, G), np.int32)
    for i in range(T - 1, -1, -1):
        tok = words[:, i][:, None]
        a, b = tok - 1, tok
        nz = tok > 0
        to_b, to_a = (pos == a) & nz, (pos == b) & nz
        moved = to_a | to_b
        pos = np.where(to_b, b, np.where(to_a, a, pos))
        hops += moved
        lo = np.minimum(lo, pos); hi = np.maximum(hi, pos)
    if check:
        assert np.array_equal(pos, apply_words_elem(words, G))
    return dict(hops=hops, width=hi - lo + 1)


def _touching(x, G=G_E):
    return {t for t in (x, x + 1) if 1 <= t <= G - 1}


def plant_elem_word(rng, k, L, w, G=G_E, N=N_E):
    """Word in which the pointer of slot k makes exactly L hops and spans exactly w adjacent positions."""
    for _ in range(200):
        lo = int(rng.integers(max(0, k - w + 1), min(k, G - w) + 1))
        hi = lo + w - 1
        e1, e2 = (lo, hi) if (k - lo) <= (hi - k) else (hi, lo)
        if abs(k - e1) + (w - 1) <= L:
            break
    else:
        raise ValueError("cannot plant")
    path, x = [k], k
    while x != e1:
        x += 1 if e1 > x else -1; path.append(x)
    while x != e2:
        x += 1 if e2 > x else -1; path.append(x)
    while len(path) - 1 < L:
        x = int(rng.choice([y for y in (x - 1, x + 1) if lo <= y <= hi])); path.append(x)
    hop_reads = np.sort(rng.choice(N, size=L, replace=False))
    hop_idx = {int(r): j for j, r in enumerate(hop_reads)}
    word, x = np.zeros(N, np.int64), k
    for r in range(N):
        t = N - 1 - r
        if r in hop_idx:
            y = path[hop_idx[r] + 1]
            word[t] = max(x, y); x = y
        else:
            bad = _touching(x, G)
            word[t] = int(rng.choice([c for c in range(G) if c not in bad]))
    return word


def plant_elem_batch(rng, B, L, w, G=G_E, N=N_E):
    words, ks = [], []
    while len(words) < B:
        k = int(rng.integers(0, G))
        try:
            words.append(plant_elem_word(rng, k, L, w, G, N)); ks.append(k)
        except ValueError:
            continue
    words, ks = np.stack(words), np.array(ks)
    tr = trajectories_elem(words, G)
    idx = np.arange(B)
    assert (tr["hops"][idx, ks] == L).all() and (tr["width"][idx, ks] == w).all(), "elementary planting failed"
    return words, ks
