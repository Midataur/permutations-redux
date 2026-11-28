"""
Cross-model probes between the three general models (torn-bigger-13, g_m10, g_plant10) on the same natural S_25 test words
(25 letters uniform over the 625 tokens).  All positions align directly: letters 0..24; START and the outputs holding
p_0..p_23 (25..49).  Method as in cross_probe.py: per pair of layers, ridge regression with position means removed, fitted on
80% of the words, held-out R^2; controls: shuffled pairing; the letters by relative offset (letter positions); the token fed in
and the answer (output positions).
Usage: python cross_probe_general.py N_WORDS -> cross_probe_general.json
"""
import sys, json, itertools
import numpy as np
import np_transformer
from np_transformer import NPTransformer
from permdata import apply_words
_orig = np_transformer.load_safetensors
np_transformer.load_safetensors = lambda p: ({k.replace("_orig_mod.", ""): v for k, v in _orig(p)[0].items()}, _orig(p)[1])
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
X = "analysis-20261126-121524/_unzipped/hpc-run-20261126-121524-extras"
MODELS = {"torn-bigger-13": "../scaling-generator/trained-models/torn-bigger-13/model.safetensors",
          "g_m10": f"{X}/g_m10/ckpt/step_00200000.safetensors", "g_plant10": f"{X}/g_plant10/ckpt/step_00200000.safetensors"}
rng = np.random.default_rng(7)
W = rng.integers(0, 625, size=(NW, 25)); P = apply_words(W, 25)
ntr = int(0.8 * NW)
H, acc = {}, {}
for name, path in MODELS.items():
    m = NPTransformer(path, 25, 25, n_blocks=5); hs, pr = [], []
    for s in range(0, NW, 100):
        toks = np.concatenate([W[s:s+100], np.full((len(W[s:s+100]), 1), m.START), P[s:s+100, :-1] + m.num_trans], 1)
        lg, hid = m.forward(toks, return_hidden=True)
        pr.append(lg.argmax(-1) - m.num_trans); hs.append(np.stack(hid, 0).astype(np.float32))
    H[name] = np.concatenate(hs, 1); acc[name] = float((np.concatenate(pr) == P).mean())
    print(name, "slot accuracy", round(acc[name], 4), flush=True)
L = 6
def ridge_r2(Xtr, Ytr, Xte, Yte, lams=(1e-2, 1e-1, 1, 10, 100)):
    """fit on half of the training words for the penalty, refit on all, score R^2 on the test words"""
    best = None
    k = len(Xtr) // 2
    for lam in lams:
        A = Xtr[:k]; Wt = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) * 1e-3, A.T @ Ytr[:k])
        r = 1 - ((Xtr[k:] @ Wt - Ytr[k:]) ** 2).sum() / ((Ytr[k:] - Ytr[k:].mean(0)) ** 2).sum()
        if best is None or r > best[0]:
            best = (r, lam)
    lam = best[1]
    Wt = np.linalg.solve(Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]) * 1e-3, Xtr.T @ Ytr)
    return float(1 - ((Xte @ Wt - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())



def flat(Z):
    return Z.reshape(-1, Z.shape[-1])

oh = np.zeros((NW, 25, 625)); oh[np.arange(NW)[:, None], np.arange(25)[None], W] = 1
rel = np.zeros((NW, 25, 49 * 2 * 25)) if ONLY != "outputs" else None      # by offset: the two points of each letter (unordered via both coords), 25 points each
pts = np.zeros((NW, 25, 50)); a, b = W // 25, W % 25
pts[np.arange(NW)[:, None], np.arange(25)[None], a] = 1; pts[np.arange(NW)[:, None], np.arange(25)[None], 25 + b] = 1
for t in range(25 if rel is not None else 0):
    for o in range(-24, 25):
        if 0 <= t + o < 25:
            rel[:, t, (o + 24) * 50:(o + 25) * 50] = pts[:, t + o]
P1 = np.eye(25)[P]; fed = np.concatenate([np.zeros((NW, 1, 25)), P1[:, :24]], 1)
CTRL = {"letters": ("letters_by_offset", rel), "outputs": ("fed_and_answer", np.concatenate([fed, P1], 2))}
groups = {"letters": np.arange(25), "outputs": np.arange(25, 50)}
perm = rng.permutation(NW)
out = dict(n_words=NW, slot_accuracy=acc, pairs={}, controls={})
for g, pos in groups.items():
    if ONLY and g != ONLY:
        continue
    C = {n: [H[n][l][:, pos] - H[n][l][:ntr, pos].mean(0, keepdims=True) for l in range(L)] for n in MODELS}
    cname, Xc = CTRL[g]; Xc = Xc - Xc[:ntr].mean(0, keepdims=True)
    for n in MODELS:
        out["controls"].setdefault(g, {})[n] = [ridge_r2(flat(Xc[:ntr]), flat(C[n][l][:ntr]), flat(Xc[ntr:]), flat(C[n][l][ntr:])) for l in range(L)]
        print(g, cname, "->", n, np.round(out["controls"][g][n], 3), flush=True)
    for A, B in itertools.permutations(MODELS, 2):
        M = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                M[i, j] = ridge_r2(flat(C[A][i][:ntr]), flat(C[B][j][:ntr]), flat(C[A][i][ntr:]), flat(C[B][j][ntr:]))
        sh = [ridge_r2(flat(C[A][i][perm][:ntr]), flat(C[B][i][:ntr]), flat(C[A][i][perm][ntr:]), flat(C[B][i][ntr:])) for i in (3, 5)]
        out["pairs"].setdefault(g, {})[f"{A} -> {B}"] = dict(matrix=M.tolist(), shuffled_layers_3_5=sh)
        print(f"\n{g}: {A} (rows) -> {B} (columns)  shuffled {np.round(sh,3)}"); print(np.array2string(M, precision=2), flush=True)
json.dump(out, open(f"cross_probe_general{'_'+ONLY if ONLY else ''}.json", "w"), indent=1)
