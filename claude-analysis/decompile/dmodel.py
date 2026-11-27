"""
PyTorch port of the two paper models (same function as np_transformer.py), plus the data they are evaluated on.

Architecture (scaling-generator/transformer.py, current version): token + learned position embeddings; n_blocks pre-LN
blocks, each  x += proj(concat_h softmax(q_h k_h^T * D^-1/2 + mask) v_h) ;  x += W2 relu(W1 LN2(x) + b1) + b2 ;
final LayerNorm (blocks.{n}) and lm_head (no bias).  q, k, v have no bias.  Mask: the N word positions attend to each
other; the output positions attend to all word positions and causally to earlier output positions.
Input: word (N letters), START, the true permutation tokens p_0..p_{n-2} (+ num_trans); the logits at the last n
positions predict p_0..p_{n-1}.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/permutations-redux/claude-analysis")
from st_loader import load_safetensors  # noqa: E402

GENERAL = dict(name="torn-bigger-13", path="/home/claude/permutations-redux/scaling-generator/trained-models/torn-bigger-13/model.safetensors",
               n=25, N=25, num_trans=625, blocks=5, heads=6)
ELEM = dict(name="elem-partitioned-long-boosted-1",
            path="/home/claude/permutations-redux/scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors",
            n=16, N=120, num_trans=16, blocks=5, heads=6)


class Weights:
    def __init__(self, spec, dtype=torch.float32):
        W, _ = load_safetensors(spec["path"])
        W = {k.replace("_orig_mod.", ""): v for k, v in W.items()}
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), dtype=dtype)
        self.spec = spec
        self.n, self.N, self.num_trans, self.L, self.H = spec["n"], spec["N"], spec["num_trans"], spec["blocks"], spec["heads"]
        self.E = t(W["token_embedding_table.weight"]); self.P = t(W["position_embedding.weight"])
        self.D = self.E.shape[1]; self.hd = self.D // self.H; self.scale = self.D ** -0.5
        self.START = self.num_trans + self.n + 1
        self.layers = []
        for i in range(self.L):
            p = f"blocks.{i}."
            g = lambda nm: torch.stack([t(W[p + f"sa.heads.{h}.{nm}.weight"]) for h in range(self.H)])   # (H, hd, D)
            self.layers.append(dict(ln1w=t(W[p + "ln1.weight"]), ln1b=t(W[p + "ln1.bias"]), ln2w=t(W[p + "ln2.weight"]), ln2b=t(W[p + "ln2.bias"]),
                                    Wq=g("query"), Wk=g("key"), Wv=g("value"),
                                    Wo=t(W[p + "sa.proj.weight"]).T.reshape(self.H, self.hd, self.D),   # head h: z_h (hd) -> D
                                    bo=t(W[p + "sa.proj.bias"]),
                                    W1=t(W[p + "ffwd.net.0.weight"]), b1=t(W[p + "ffwd.net.0.bias"]),
                                    W2=t(W[p + "ffwd.net.2.weight"]), b2=t(W[p + "ffwd.net.2.bias"])))
        self.lnfw, self.lnfb = t(W[f"blocks.{self.L}.weight"]), t(W[f"blocks.{self.L}.bias"])
        self.Wlm = t(W["lm_head.weight"])
        T = self.N + self.n
        M = torch.zeros(T, T, dtype=torch.bool)
        M[:self.N, :self.N] = True; M[self.N:, :self.N] = True
        M[self.N:, self.N:] = torch.tril(torch.ones(self.n, self.n, dtype=torch.bool))
        self.mask = M                                     # True = may attend
        self.T = T


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps) * w + b


def linear_layer_norm(x, w, b, var, eps=1e-5):
    """LayerNorm with the variance replaced by a constant (LLNA)."""
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(torch.as_tensor(var) + eps) * w + b


def tokens(m: Weights, words, perms):
    B = words.shape[0]
    start = torch.full((B, 1), m.START, dtype=torch.long)
    return torch.cat([words, start, perms[:, :-1] + m.num_trans], 1)


def attend(m, q, k, v):
    """q,k,v: (B,T,hd) for one head -> (B,T,hd)."""
    s = (q @ k.transpose(1, 2)) * m.scale
    s = s.masked_fill(~m.mask, float("-inf"))
    return torch.softmax(s, -1) @ v


def forward(m: Weights, toks, record=None):
    """Plain forward; returns logits at the n output positions (B,n,V). record: dict to store LN input variances."""
    x = m.E[toks] + m.P[: toks.shape[1]]
    for i, ly in enumerate(m.layers):
        a = layer_norm(x, ly["ln1w"], ly["ln1b"])
        if record is not None:
            record.setdefault(("ln1", i), []).append(x.var(-1, unbiased=False).mean().item())
        out = 0
        for h in range(m.H):
            z = attend(m, a @ ly["Wq"][h].T, a @ ly["Wk"][h].T, a @ ly["Wv"][h].T)
            out = out + z @ ly["Wo"][h]
        x = x + out + ly["bo"]
        if record is not None:
            record.setdefault(("ln2", i), []).append(x.var(-1, unbiased=False).mean().item())
        f = layer_norm(x, ly["ln2w"], ly["ln2b"])
        x = x + torch.relu(f @ ly["W1"].T + ly["b1"]) @ ly["W2"].T + ly["b2"]
    if record is not None:
        record.setdefault(("lnf",), []).append(x.var(-1, unbiased=False).mean().item())
    x = layer_norm(x, m.lnfw, m.lnfb)
    return x[:, -m.n:] @ m.Wlm.T


# ---------------------------------------------------------------- data
def perms_general(words, n=25):
    B = len(words); arr = np.tile(np.arange(n), (B, 1)); rows = np.arange(B)
    for i in range(words.shape[1]):
        a, b = words[:, i] // n, words[:, i] % n
        x, y = arr[rows, a].copy(), arr[rows, b].copy(); arr[rows, a], arr[rows, b] = y, x
    return arr


def perms_elem(words, n=16):
    B = len(words); arr = np.tile(np.arange(n), (B, 1)); rows = np.arange(B)
    for i in range(words.shape[1]):
        t = words[:, i]; mk = t > 0; r = rows[mk]; a, b = t[mk] - 1, t[mk]
        x, y = arr[r, a].copy(), arr[r, b].copy(); arr[r, a], arr[r, b] = y, x
    return arr


def sample_words(m: Weights, B, rng):
    """Test-distribution words: N letters uniform over all tokens (the paper's test words)."""
    w = rng.integers(0, m.num_trans, size=(B, m.N))
    p = perms_general(w, m.n) if m.num_trans == m.n ** 2 else perms_elem(w, m.n)
    return torch.tensor(w, dtype=torch.long), torch.tensor(p, dtype=torch.long)
