"""
Test C: is the learned token-embedding kernel S_n-invariant?

Tokens s_{a,b} (a<b) are indexed by 2-subsets of [n]. The S_n-permutation module on 2-subsets
splits (multiplicity-free, n>=4) as  V0 (dim 1) + V1 (dim n-1) + V2 (dim n(n-3)/2).
Any S_n-invariant Gram matrix is  G = l0*P0 + l1*P1 + l2*P2  (P_k = isotypic projectors), i.e.
it lies in the Bose-Mesner algebra of the Johnson scheme J(n,2) spanned by I, A1 (overlap 1), A2 (overlap 0).

We measure, for the trained embedding E (rows = tokens):
  (a) how far E[s_ab] is from E[s_ba]
  (b) leakage: fraction of ||G||_F^2 lying OUTSIDE the three diagonal blocks  P_k G P_k
  (c) within-block eigenvalue spread (should be ~0 for an invariant kernel)
  (d) fraction of off-diagonal Gram energy captured by the 3-class means (I, A1, A2 fit)
and compare each with a null where the 300 tokens are randomly relabelled (destroying any S_n structure).
"""
from paths import MODEL_TORN
import itertools, json, sys
import numpy as np
from st_loader import load_safetensors

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    MODEL_TORN
n = 25
W, _ = load_safetensors(MODEL)
E = W["token_embedding_table.weight"]            # (vocab, D)
tok = lambda a, b: a * n + b

pairs = list(itertools.combinations(range(n), 2))
idx = {p: i for i, p in enumerate(pairs)}
m = len(pairs)                                    # 300

# ---- (a) s_ab vs s_ba
cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
sym = np.array([cos(E[tok(a, b)], E[tok(b, a)]) for a, b in pairs])
print(f"(a) cos(E[s_ab], E[s_ba]) over {m} pairs: mean={sym.mean():.4f} min={sym.min():.4f}")

# Use the symmetrised embedding for the S_n analysis
X = np.stack([(E[tok(a, b)] + E[tok(b, a)]) / 2 for a, b in pairs])      # (300, D)
G = X @ X.T

# ---- isotypic projectors on R^{pairs}
inc = np.zeros((m, n))                            # incidence: pair -> its two points
for i, (a, b) in enumerate(pairs):
    inc[i, a] = inc[i, b] = 1
P0 = np.ones((m, m)) / m
U, S, _ = np.linalg.svd(inc, full_matrices=False)  # column space of incidence = V0 + V1 (dim n)
Q01 = U[:, S > 1e-9]
P01 = Q01 @ Q01.T
P1 = P01 - P0
P2 = np.eye(m) - P01
dims = [int(round(np.trace(P))) for P in (P0, P1, P2)]
print(f"    isotypic dims: {dims}  (expected 1, {n-1}, {n*(n-3)//2})")


def stats(G):
    Ps = (P0, P1, P2)
    tot = np.sum(G ** 2)
    diag_blocks = sum(np.sum((P @ G @ P) ** 2) for P in Ps)
    leak = 1 - diag_blocks / tot
    spread = []
    lam = []
    for P, d in zip(Ps, dims):
        ev = np.linalg.eigvalsh(P @ G @ P)
        ev = np.sort(ev)[-d:]                      # the d eigenvalues living on this block
        lam.append(float(ev.mean()))
        spread.append(float(ev.std() / abs(ev.mean())))
    # class-mean fit of the Bose-Mesner algebra on the off-diagonal, I, A1, A2
    A1 = (inc @ inc.T == 1).astype(float)
    A2 = (inc @ inc.T == 0).astype(float)
    A0 = np.eye(m)
    fit = sum(np.sum(G * A) / np.sum(A) * A for A in (A0, A1, A2))
    off = ~np.eye(m, dtype=bool)
    Goff = G - np.diag(np.diag(G))
    resid = np.sum((Goff - (fit - np.diag(np.diag(fit)))) ** 2) / np.sum(Goff ** 2)
    return dict(leakage=float(leak), eig_means=lam, eig_rel_spread=spread,
                offdiag_unexplained=float(resid),
                class_means=[float(np.sum(G * A) / np.sum(A)) for A in (A0, A1, A2)])


res = stats(G)
print("(b-d) trained embedding:")
print(f"    leakage outside diagonal isotypic blocks : {res['leakage']:.4f}")
print(f"    block eigenvalues (V0,V1,V2)             : {[round(x,3) for x in res['eig_means']]}")
print(f"    within-block relative eigenvalue spread  : {[round(x,4) for x in res['eig_rel_spread']]}")
print(f"    off-diagonal energy NOT explained by 3 class means : {res['offdiag_unexplained']:.4f}")
print(f"    class means (diag, overlap-1, disjoint)  : {[round(x,3) for x in res['class_means']]}")

# ---- null: relabel the 300 tokens at random (not induced by any S_n element)
rng = np.random.default_rng(0)
nulls = []
for _ in range(20):
    pi = rng.permutation(m)
    nulls.append(stats(G[np.ix_(pi, pi)]))
for key in ("leakage", "offdiag_unexplained"):
    v = np.array([x[key] for x in nulls])
    print(f"null (random relabelling) {key}: mean={v.mean():.4f}  min={v.min():.4f}")
sp = np.array([x["eig_rel_spread"] for x in nulls])
print(f"null within-block rel. spread (V0,V1,V2): {np.round(sp.mean(0),4)}")

# ---- permutation tokens and identity (diagonal) tokens: should be aI + b(J-I) if S_n-structured
ntr = n * n
for name, rows in (("permutation tokens", [ntr + i for i in range(n)]),
                   ("identity tokens s_aa", [tok(a, a) for a in range(n)])):
    Xp = E[rows]
    Gp = Xp @ Xp.T
    d = np.diag(Gp)
    off = Gp[~np.eye(n, dtype=bool)]
    ev = np.sort(np.linalg.eigvalsh(Gp))[::-1]
    print(f"{name}: diag mean={d.mean():.3f} (sd {d.std():.3f}); off-diag mean={off.mean():.3f} (sd {off.std():.3f}); "
          f"top eig={ev[0]:.2f}; next eigs mean={ev[1:].mean():.3f} sd={ev[1:].std():.3f}")

json.dump(dict(trained=res, null=[{k: v for k, v in x.items()} for x in nulls[:3]]),
          open("gram_results.json", "w"), indent=1)

# ---- spectrum figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    evn = np.sort(np.linalg.eigvalsh(G[np.ix_(pi, pi)]))[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ev, ".", label="trained embedding Gram (300 pair-tokens)")
    ax.plot(evn, "x", alpha=.4, label="randomly relabelled (same spectrum)")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("eigenvalue"); ax.legend(); ax.set_yscale("symlog")
    fig.tight_layout(); fig.savefig("gram_spectrum.png", dpi=130)
except Exception as e:
    print("figure skipped:", e)


# ---- follow-up: is the token embedding (mostly) a sum of per-point embeddings?
# Additive models fitted by least squares on the 600 ordered off-diagonal tokens (a != b), all D dims at once.
print("\n(e) additive-structure test on ordered tokens s_(a,b), a!=b:")
ordered = [(a, b) for a in range(n) for b in range(n) if a != b]
Y = np.stack([E[tok(a, b)] for a, b in ordered])                 # (600, D)
Yc = Y - Y.mean(0, keepdims=True)
tot = np.sum(Yc ** 2)


def design(kind):
    rows = []
    for a, b in ordered:
        r = np.zeros(2 * n if kind == "separate" else n)
        if kind == "separate":
            r[a] = 1; r[n + b] = 1           # f_a + g_b (first slot / second slot features)
        else:
            r[a] += 1; r[b] += 1             # u_a + u_b (shared point features, symmetric)
        rows.append(r)
    return np.array(rows)


for kind in ("separate", "shared"):
    Xd = design(kind)
    Xd1 = np.hstack([Xd, np.ones((len(ordered), 1))])
    coef, *_ = np.linalg.lstsq(Xd1, Yc, rcond=None)
    r2 = 1 - np.sum((Yc - Xd1 @ coef) ** 2) / tot
    print(f"    R^2, E[s_ab] ~ {'f_a + g_b' if kind=='separate' else 'u_a + u_b'} : {r2:.3f}"
          f"   (params/dim = {np.linalg.matrix_rank(Xd1)}; chance R^2 ~ {np.linalg.matrix_rank(Xd1)/len(ordered):.3f})")
# energy of the symmetrised embedding in V0+V1 (additive) vs V2 (pair-specific), as fractions of trace
trG = np.trace(G)
print(f"    symmetrised embedding energy: V0+V1 (additive) = {np.trace(P01 @ G)/trG:.3f}, V2 (pair-specific) = {np.trace(P2 @ G)/trG:.3f}")
