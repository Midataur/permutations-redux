"""Coverage scaling law (section 3.27): which training group S_n covers which test group S_N.

Test words: m_te letters, each uniform over the N^2 tokens (a,b) of [N]^2 (a letter (a,b) with a != b swaps a and b).
Training words: m_tr letters uniform over the n^2 tokens of S_n, relabelled into a random n-subset of [N].
A slot's chain visits s points; it is covered by the training distribution iff s <= n.

Exact: the pointer hops at each letter with probability p_N = 2(N-1)/N^2, to a uniform other point, so the number of
points visited is a Markov chain.  Limit: N -> infinity at fixed lambda = 2 m_te / N gives s - 1 ~ Poisson(lambda).
"""
import json, math, sys
import numpy as np
from scipy.stats import poisson


def points_dist(N, m):
    """P(s = j) for j = 1..min(N, m+1): number of distinct points a slot's chain visits, test words of m letters on N points."""
    p = 2 * (N - 1) / N**2
    d = np.zeros(min(N, m + 1) + 1); d[1] = 1.0
    for _ in range(m):
        nd = d * (1 - p)
        for s in range(1, len(d)):
            if d[s] == 0:
                continue
            new = (N - s) / (N - 1)
            nd[s] += d[s] * p * (1 - new)
            if s + 1 < len(d):
                nd[s + 1] += d[s] * p * new
        d = nd
    return d


def uncovered(n, N, m):
    d = points_dist(N, m); return float(d[n + 1:].sum())


def n_star(N, m, eps):
    d = points_dist(N, m); tail = np.cumsum(d[::-1])[::-1]   # tail[j] = P(s >= j)
    for n in range(1, len(d)):
        if n + 1 >= len(d) or tail[n + 1] <= eps:
            return n
    return len(d) - 1


def n_star_limit(lam, eps):
    n = 1
    while poisson.sf(n - 1, lam) > eps:   # P(L >= n) = P(s >= n+1)
        n += 1
    return n


def ratio(L, s, n, N, m_tr, m_te):
    """Training/test probability of one labelled walk with L hops through s points (positions of the hops summed over)."""
    if s > n:
        return 0.0
    ff = math.prod((n - i) / (N - i) for i in range(s))
    qn, qN = ((n - 1)**2 + 1) / n**2, ((N - 1)**2 + 1) / N**2
    return ff * math.comb(m_tr, L) / math.comb(m_te, L) * (N / n)**(2 * L) * qn**(m_tr - L) / qN**(m_te - L)


if __name__ == "__main__":
    out = {}
    # 1. checks against section 3.19 (n = 10, N = 25, m = 25)
    out["check_uncovered_10_25_25"] = uncovered(10, 25, 25)                    # simulation: 30 in 25M = 1.2e-6
    out["check_R"] = {f"{L},{s}": ratio(L, s, 10, 25, 25, 25) for L, s in [(1, 2), (2, 3), (8, 9)]}   # 0.055, 0.133, 1.519
    # 2. n* against N, test length proportional to N (fixed lambda) and fixed
    Ns = [10, 16, 25, 50, 100, 250, 1000]
    out["n_star"] = {}
    for eps in (1e-2, 1e-4, 1e-6):
        for lab, mfun in (("m=N", lambda N: N), ("m=2N", lambda N: 2 * N), ("m=25", lambda N: 25)):
            out["n_star"][f"{lab}, eps={eps:g}"] = {N: n_star(N, mfun(N), eps) for N in Ns}
    out["n_star_limit"] = {f"lambda={lam}, eps={eps:g}": n_star_limit(lam, eps) for lam in (1, 2, 4, 8) for eps in (1e-2, 1e-4, 1e-6)}
    # 3. uncovered mass for the paper's pair and a few others
    out["uncovered"] = {f"n={n}, N={N}, m={m}": uncovered(n, N, m) for n, N, m in
                        [(10, 25, 25), (10, 50, 50), (10, 100, 100), (10, 1000, 1000), (10, 100, 25), (10, 25, 50), (16, 25, 25), (6, 25, 25)]}
    # 4. min ratio over classes with test mass, at large N: R_min ~ (n/N) e^{lambda_te - lambda_tr}
    out["R_min_examples"] = {f"n={n}, N={N}, m_tr={mt}, m_te={me}": min(ratio(L, L + 1, n, N, mt, me) for L in range(0, 4))
                             for n, N, mt, me in [(10, 25, 25, 25), (10, 100, 25, 100), (10, 1000, 25, 1000)]}
    json.dump(out, open("scaling_law.json", "w"), indent=1); print(json.dumps(out, indent=1))
