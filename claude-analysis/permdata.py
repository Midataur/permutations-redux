"""
Data utilities for the general-transposition model (token = a*n + b encodes the transposition
s_{a,b}; a==b is the identity), mirroring fast-data-gen/src/main.rs:
  * generate_random_sequence : each letter uniform on [0, n^2)
  * get_permutation          : start from the identity arrangement and swap POSITIONS a,b per letter
"""
import numpy as np


def random_words(rng, B, n=25, N=25):
    return rng.integers(0, n * n, size=(B, N))


def apply_words(words, n=25):
    """Arrangement after applying each word's swaps in order (same as get_permutation)."""
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
    """Pointer chase for every output position k, reading the word right to left.
    Returns dict with (B,n) arrays:
        hops     number of letters that moved the pointer
        distinct number of distinct points visited (including the start)
        hopmask  (B,n,N) bool, True where letter t moved pointer k
    and asserts that the pointer's final position equals the true permutation entry p_k."""
    B, N = words.shape
    pos = np.tile(np.arange(n), (B, 1))
    hops = np.zeros((B, n), dtype=np.int32)
    visited = np.zeros((B, n, n), dtype=bool)
    visited[:, np.arange(n), np.arange(n)] = True
    hopmask = np.zeros((B, n, N), dtype=bool)
    for t in range(N - 1, -1, -1):
        a, b = (words[:, t] // n)[:, None], (words[:, t] % n)[:, None]
        nz = a != b
        to_b, to_a = (pos == a) & nz, (pos == b) & nz
        moved = to_a | to_b
        new = np.where(to_b, b, np.where(to_a, a, pos))
        hops += moved
        hopmask[:, :, t] = moved
        pos = new
        bi, ki = np.nonzero(moved)
        visited[bi, ki, pos[bi, ki]] = True
    out = dict(hops=hops, distinct=visited.sum(2), hopmask=hopmask, final=pos)
    if check:
        assert np.array_equal(pos, apply_words(words, n)), "pointer picture disagrees with permutation"
    return out


def moved_points(perm):
    """Number of non-fixed points of each output permutation."""
    return (perm != np.arange(perm.shape[1])[None]).sum(1)


# ---------------------------------------------------------------------------------------
# Planted trajectories ("ladder words")
# ---------------------------------------------------------------------------------------
def plant_word(rng, k, L, s, n=25, N=25, hop_reads=None):
    """Build one word in which the pointer of output position k makes exactly L hops and visits
    exactly s distinct points (including its start); every non-hop letter is random but avoids
    the pointer's current position, so it cannot move it.

    hop_reads : optional sorted array of L reading-order indices (0 = LAST letter of the word,
                since the pointer chase reads right to left) at which the pointer hops.
    Requires 2 <= s <= min(L+1, n)."""
    assert 2 <= s <= min(L + 1, n) and L <= N
    if hop_reads is None:
        hop_reads = np.sort(rng.choice(N, size=L, replace=False))
    hop_set = set(int(r) for r in hop_reads)
    new_flag = np.zeros(L, dtype=bool)
    new_flag[0] = True                                   # first hop must leave the start point
    if s > 2:
        new_flag[rng.choice(np.arange(1, L), size=s - 2, replace=False)] = True
    x, visited, word, h = k, [k], np.zeros(N, dtype=np.int64), 0
    for r in range(N):
        t = N - 1 - r
        if r in hop_set:
            if new_flag[h]:
                y = int(rng.choice([p for p in range(n) if p not in visited]))
                visited.append(y)
            else:
                y = int(rng.choice([p for p in visited if p != x]))
            a, b = (x, y) if rng.random() < 0.5 else (y, x)
            word[t] = a * n + b
            x, h = y, h + 1
        else:
            others = [p for p in range(n) if p != x]
            a, b = rng.choice(others), rng.choice(others)
            word[t] = a * n + b
    return word


def plant_batch(rng, B, L, s, n=25, N=25, hop_reads_fn=None):
    ks = rng.integers(0, n, size=B)
    words = np.stack([plant_word(rng, int(k), L, s, n, N,
                                 None if hop_reads_fn is None else hop_reads_fn(rng, L, N))
                      for k in ks])
    tr = trajectories(words, n)
    idx = np.arange(B)
    assert (tr["hops"][idx, ks] == L).all() and (tr["distinct"][idx, ks] == s).all(), "planting failed"
    return words, ks


def pointer_path(word, k, n=25):
    """Positions of the pointer for output slot k after 0,1,2,... hops (reading right to left)."""
    x, path = int(k), [int(k)]
    for t in range(len(word) - 1, -1, -1):
        a, b = int(word[t]) // n, int(word[t]) % n
        if a != b and x in (a, b):
            x = b if x == a else a
            path.append(x)
    return path
