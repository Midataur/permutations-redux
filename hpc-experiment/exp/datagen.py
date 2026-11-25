"""On-the-fly training-data generators, written in torch so that they run on the GPU next to the model.

Nothing is read from disk: every optimiser step gets a fresh batch (no epochs, so no memorisation).  The distributions
follow the repo's Rust generator (fast-data-gen/src/main.rs) and the original data statistics (see tests/).

General problem (n=25 points, 25 letters, token = n*a+b):
  * "natural" word over m points: letters uniform over the m^2 ordered pairs (a == b is the identity), then the m points
    are relabelled by a uniformly random permutation of the 25 labels  (== `-g m -M 25 -S -R` in the Rust generator).
  * "planted" word: one output slot k is given a chain of exactly L hops through exactly s distinct points, all letters
    on a random m-point set; every non-hop letter avoids the pointer, so it cannot disturb the chain.

Elementary problem (16 points, 120 letters): the repo's partitioned-window generator (`-k -p`): a random partition of 10
into parts >= 3 gives window sizes; each window gets a random shift; each letter picks a random window and a value
i ~ U{0..9}, and the token is shift + (i mod size).
"""
import itertools
import torch

# ---------------------------------------------------------------------------------------------- shared
def apply_swaps_general(words, n=25):
    """Arrangement after swapping positions (a, b) for every letter in order. words: (B, N) long."""
    B, N = words.shape
    perm = torch.arange(n, device=words.device).expand(B, n).clone()
    a, b = words // n, words % n
    for t in range(N):
        ia, ib = a[:, t:t + 1], b[:, t:t + 1]
        va, vb = perm.gather(1, ia), perm.gather(1, ib)
        perm.scatter_(1, ia, vb)
        perm.scatter_(1, ib, va)
    return perm


def apply_swaps_elem(words, G=16):
    B, N = words.shape
    arr = torch.arange(G, device=words.device).expand(B, G).clone()
    for i in range(N):
        t = words[:, i:i + 1]
        nz = t > 0
        ia, ib = torch.where(nz, t - 1, torch.zeros_like(t)), torch.where(nz, t, torch.zeros_like(t))
        va, vb = arr.gather(1, ia), arr.gather(1, ib)
        arr.scatter_(1, ia, vb)
        arr.scatter_(1, ib, va)
    return arr


# ---------------------------------------------------------------------------------------------- general
class GeneralSampler:
    """dist = dict(ms={m: prob, ...}, planted_frac=0.0, plant_m=10, Lmin=3, Lmax=25)."""
    n, N = 25, 25

    def __init__(self, device, dist, seed):
        self.dev = torch.device(device)
        self.dist = dict(dist)
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(seed)
        ms = sorted(self.dist["ms"].items())
        self.m_vals = torch.tensor([m for m, _ in ms], device=self.dev)
        self.m_probs = torch.tensor([p for _, p in ms], device=self.dev, dtype=torch.float32)
        self.m_probs = self.m_probs / self.m_probs.sum()

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def randint(self, lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=self.g, device=self.dev)

    def natural(self, B, m_vec=None):
        n, N = self.n, self.N
        if m_vec is None:
            m_vec = self.m_vals[torch.multinomial(self.m_probs, B, replacement=True, generator=self.g)]
        mm = m_vec[:, None]
        a = torch.minimum((self.rand(B, N) * mm).long(), mm - 1)
        b = torch.minimum((self.rand(B, N) * mm).long(), mm - 1)
        rel = self.rand(B, n).argsort(1)
        return rel.gather(1, a) * n + rel.gather(1, b)

    def planted(self, B, m, Lmin, Lmax):
        """Returns words (B,N), k_label (B,), L (B,), s (B,)."""
        n, N, dev = self.n, self.N, self.dev
        ar = torch.arange(B, device=dev)
        L = self.randint(Lmin, Lmax + 1, (B,))
        smax = torch.clamp(L + 1, max=m)
        s = 2 + torch.minimum((self.rand(B) * (smax - 1)).long(), smax - 2)      # uniform on 2..smax
        rel = self.rand(B, n).argsort(1)
        k = self.randint(0, m, (B,))
        ranks = self.rand(B, N).argsort(1).argsort(1)
        is_hop = ranks < L[:, None]                                              # exactly L of the N reading steps
        hop_idx = (is_hop.long().cumsum(1) - 1).clamp(min=0)
        sc = self.rand(B, N)
        sc[:, 0] = 2.0                                                           # the first hop must leave the start
        sc = torch.where(torch.arange(N, device=dev)[None, :] < L[:, None], sc, torch.full_like(sc, -1.0))
        new_flag = sc.argsort(1, descending=True).argsort(1) < (s - 1)[:, None]  # which hops visit a NEW point
        x = k.clone()
        visited = torch.zeros(B, m, dtype=torch.bool, device=dev)
        visited[ar, k] = True
        cols = torch.arange(m, device=dev)[None, :]
        letters = torch.zeros(B, N, dtype=torch.long, device=dev)
        for r in range(N):
            hop = is_hop[:, r]
            isnew = new_flag.gather(1, hop_idx[:, r:r + 1])[:, 0]
            y_new = torch.where(~visited, self.rand(B, m), torch.full((B, m), -1.0, device=dev)).argmax(1)
            y_old = torch.where(visited & (cols != x[:, None]), self.rand(B, m), torch.full((B, m), -1.0, device=dev)).argmax(1)
            y = torch.where(isnew, y_new, y_old)
            u = torch.minimum((self.rand(B) * (m - 1)).long(), torch.full_like(x, m - 2))
            v = torch.minimum((self.rand(B) * (m - 1)).long(), torch.full_like(x, m - 2))
            u, v = u + (u >= x).long(), v + (v >= x).long()                      # uniform over points != x
            coin = self.rand(B) < 0.5
            a = torch.where(hop, torch.where(coin, x, y), u)
            b = torch.where(hop, torch.where(coin, y, x), v)
            letters[:, N - 1 - r] = rel.gather(1, a[:, None])[:, 0] * n + rel.gather(1, b[:, None])[:, 0]
            visited = visited | (hop[:, None] & (cols == y[:, None]))
            x = torch.where(hop, y, x)
        return letters, rel.gather(1, k[:, None])[:, 0], L, s

    def sample(self, B):
        """Returns words (B,25), perm (B,25)."""
        d = self.dist
        n_pl = int(round(B * d.get("planted_frac", 0.0)))
        parts = []
        if B - n_pl > 0:
            parts.append(self.natural(B - n_pl))
        if n_pl > 0:
            parts.append(self.planted(n_pl, d["plant_m"], d["Lmin"], d["Lmax"])[0])
        words = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
        return words, apply_swaps_general(words, self.n)


def general_batch(words, perm, n=25):
    """Model input and target tokens (masked-model layout of the repo): word | START | p_0..p_{n-2};  target p_0..p_{n-1}."""
    B = words.shape[0]
    START = n * n + n + 1
    start = torch.full((B, 1), START, dtype=torch.long, device=words.device)
    inp = torch.cat([words, start, perm[:, :-1] + n * n], 1)
    return inp, perm + n * n


# ---------------------------------------------------------------------------------------------- elementary
def _partitions(total, min_part):
    out = []
    def rec(rem, mx, cur):
        if rem == 0:
            out.append(tuple(cur)); return
        for p in range(min(rem, mx), min_part - 1, -1):
            rec(rem - p, p, cur + [p])
    rec(total, total, [])
    return out


ELEM_PARTITIONS = _partitions(10, 3)        # [(10,), (7,3), (6,4), (5,5), (4,3,3)]  -- all equally likely (Fristedt)


class ElemSampler:
    G, N = 16, 120

    def __init__(self, device, seed, group_size=10):
        self.dev = torch.device(device)
        self.g = torch.Generator(device=self.dev); self.g.manual_seed(seed)
        parts = _partitions(group_size, 3)
        k = max(len(p) for p in parts)
        self.sizes = torch.tensor([list(p) + [1] * (k - len(p)) for p in parts], device=self.dev)
        self.nparts = torch.tensor([len(p) for p in parts], device=self.dev)
        self.group_size = group_size

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def words(self, B):
        G, N = self.G, self.N
        pid = torch.randint(0, len(self.nparts), (B,), generator=self.g, device=self.dev)
        sz, npart = self.sizes[pid], self.nparts[pid]
        shift = torch.minimum((self.rand(B, sz.shape[1]) * (G - sz + 1)).long(), G - sz)
        j = torch.minimum((self.rand(B, N) * npart[:, None]).long(), npart[:, None] - 1)
        i = torch.minimum((self.rand(B, N) * self.group_size).long(), torch.full((B, N), self.group_size - 1, device=self.dev))
        return shift.gather(1, j) + (i % sz.gather(1, j))

    def sample(self, B):
        w = self.words(B)
        return w, apply_swaps_elem(w, self.G)


def elem_batch(words, perm, G=16):
    B = words.shape[0]
    START = 2 * G + 1
    start = torch.full((B, 1), START, dtype=torch.long, device=words.device)
    return torch.cat([words, start, perm[:, :-1] + G], 1), perm + G
