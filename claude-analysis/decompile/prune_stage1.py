"""
Stage 1 of the decompilation pipeline of Huang et al. (arXiv:2602.08857; code lacoco-lab/decompiling_transformers,
pruning_model.py: MaskSampler + PruningModelWithHooks), ported to the architecture of the two paper models.

Graph.  Sources: tok, pos, every attention head's output a{l}.{h} (its share of the attention projection, without the
projection bias), every MLP output m{l}.  Destinations: the query, key and value input of every head, every MLP input,
and the unembedding (lm).  A destination at layer l reads the sources before it (heads of the same layer feed the MLP).
Every (destination, source) edge has a gate.  A closed gate replaces the source by a learned constant vector (optimal
ablation); every destination also gets a learned constant (initialised with the attention projection biases it would
have received).  LayerNorm: either the real one (LN=real) or the linear version (LN=lin): (x - mean)/sqrt(c) * w + b
with learned constants c, one per destination and position (the per-position version of the paper's LLNA; one per
destination fails for these models, see llna_fit.py).

Training: KL divergence to the original model at the output positions (summed over the vocabulary, averaged over
positions) + lambda * sum(sigmoid(gate parameters)); gate sampling and dangling-edge removal as in the original code.
Evaluation: gates thresholded at 0, dangling edges removed; match accuracy = share of fresh test words whose argmax
prediction equals the original model's at every output position (teacher-forced with the true permutation).

Usage: python prune_stage1.py MODEL(general|elem) LAMBDA LN(lin|real) STEPS OUT.json [UNIQ REPEAT]
"""
import sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from dmodel import *

which, lamb, lnmode, STEPS, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
UNIQ = int(sys.argv[6]) if len(sys.argv) > 6 else 6
REP = int(sys.argv[7]) if len(sys.argv) > 7 else 6
torch.manual_seed(0); rng = np.random.default_rng(0)
_X = "/home/claude/permutations-redux/claude-analysis/analysis-20261126-121524/_unzipped/hpc-run-20261126-121524-extras"
SPECS = {"general": GENERAL, "elem": ELEM,
         "g_m10": dict(GENERAL, name="g_m10", path=f"{_X}/g_m10/ckpt/step_00200000.safetensors"),
         "g_plant10": dict(GENERAL, name="g_plant10", path=f"{_X}/g_plant10/ckpt/step_00200000.safetensors")}
m = Weights(SPECS[which])
L, H, D, T = m.L, m.H, m.D, m.T

# ------------------------------------------------------------------ graph
# Sources in computation order: tok, pos, a0.0..a0.5, m0, a1.0.., m1, ...  A destination reads a prefix of this list.
SRC = ["tok", "pos"]
for l in range(L):
    SRC += [f"a{l}.{h}" for h in range(H)] + [f"m{l}"]
sid = {s: i for i, s in enumerate(SRC)}
NS = len(SRC)


def n_before(l, mlp=False):
    return 2 + l * (H + 1) + (H if mlp else 0)


DEST = [(l, act, h) for l in range(L) for act in ("q", "k", "v") for h in range(H)] + [(l, "mlp") for l in range(L)] + [("lm",)]
did = {d: i for i, d in enumerate(DEST)}
dest_n = {d: (NS if d[0] == "lm" else n_before(d[0], d[1] == "mlp")) for d in DEST}
EDGES = [(d, SRC[j]) for d in DEST for j in range(dest_n[d])]
eid = {e: i for i, e in enumerate(EDGES)}
E = len(EDGES)
dest_srcs = {d: SRC[:dest_n[d]] for d in DEST}
# dense (dest, source) -> edge index, -1 where there is no edge
EIDX = torch.full((len(DEST), NS), -1, dtype=torch.long)
for (d, s), i in eid.items():
    EIDX[did[d], sid[s]] = i
OUTNODE = {d: ("lm" if d[0] == "lm" else (f"m{d[0]}" if d[1] == "mlp" else f"a{d[0]}.{d[2]}")) for d in DEST}


def ln_dest(d):
    return "lnf" if d[0] == "lm" else (f"ln2.{d[0]}" if d[1] == "mlp" else f"ln1.{d[0]}")


# ------------------------------------------------------------------ parameters
INIT = 8.0
gate = torch.nn.Parameter(torch.full((E,), INIT))
oa_in = torch.nn.Parameter(torch.zeros(NS, D))
oa_out = torch.nn.Parameter(torch.zeros(len(DEST), D))
with torch.no_grad():
    for d in DEST:
        upto = L if d[0] == "lm" else (d[0] + 1 if d[1] == "mlp" else d[0])
        for k in range(upto):
            oa_out[did[d]] += m.layers[k]["bo"]


def ln_stats():
    st = {}
    with torch.no_grad():
        w, p = sample_words(m, 256, np.random.default_rng(99)); toks = tokens(m, w, p)
        x = m.E[toks] + m.P[:T]
        for i, ly in enumerate(m.layers):
            st[f"ln1.{i}"] = x.var(-1, unbiased=False).mean(0)
            a = layer_norm(x, ly["ln1w"], ly["ln1b"]); out = 0
            for h in range(H):
                out = out + attend(m, a @ ly["Wq"][h].T, a @ ly["Wk"][h].T, a @ ly["Wv"][h].T) @ ly["Wo"][h]
            x = x + out + ly["bo"]
            st[f"ln2.{i}"] = x.var(-1, unbiased=False).mean(0)
            f = layer_norm(x, ly["ln2w"], ly["ln2b"]); x = x + torch.relu(f @ ly["W1"].T + ly["b1"]) @ ly["W2"].T + ly["b2"]
        st["lnf"] = x.var(-1, unbiased=False).mean(0)
    return st


st = ln_stats()
logvar = torch.nn.Parameter(torch.stack([st[ln_dest(d)].log() for d in DEST]))       # (n_dest, T)
LNW = {}
for d in DEST:
    if d[0] == "lm":
        LNW[d] = (m.lnfw, m.lnfb)
    else:
        ly = m.layers[d[0]]
        LNW[d] = (ly["ln2w"], ly["ln2b"]) if d[1] == "mlp" else (ly["ln1w"], ly["ln1b"])


def norm(x, dests):
    """x: (B, k, T, D) inputs of the destinations in `dests` (all of one LN type)."""
    w = torch.stack([LNW[d][0] for d in dests]).view(1, len(dests), 1, D)
    b = torch.stack([LNW[d][1] for d in dests]).view(1, len(dests), 1, D)
    if lnmode == "real":
        mu = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-5) * w + b
    v = logvar[[did[d] for d in dests]].exp().view(1, len(dests), T, 1)
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(v + 1e-5) * w + b


# ------------------------------------------------------------------ masks (MaskSampler of the original code)
def prune_dangling(masks):
    B = masks.shape[0]
    with torch.no_grad():
        on = masks > 0
        def edge_on(d):       # (B, n) for the prefix sources of d
            return on[:, EIDX[did[d], :dest_n[d]]]
        reach_in = torch.zeros(B, NS, dtype=torch.bool); reach_in[:, :2] = True
        for l in range(L):
            for h in range(H):
                r = torch.ones(B, dtype=torch.bool)
                for act in ("q", "k", "v"):
                    d = (l, act, h); r &= (edge_on(d) & reach_in[:, :dest_n[d]]).any(1)
                reach_in[:, sid[f"a{l}.{h}"]] = r
            d = (l, "mlp"); reach_in[:, sid[f"m{l}"]] = (edge_on(d) & reach_in[:, :dest_n[d]]).any(1)
        reach_out = torch.zeros(B, NS, dtype=torch.bool)
        order = [("lm",)] + [x for l in range(L - 1, -1, -1) for x in ([(l, "mlp")] + [(l, a, h) for a in ("q", "k", "v") for h in range(H)])]
        for d in order:
            n = dest_n[d]; o = OUTNODE[d]
            ro = torch.ones(B, dtype=torch.bool) if o == "lm" else (reach_out[:, sid[o]] & reach_in[:, sid[o]])
            reach_out[:, :n] |= edge_on(d) & ro.unsqueeze(1) & reach_in[:, :n]
        keep = torch.zeros_like(on)
        for d in DEST:
            n = dest_n[d]; o = OUTNODE[d]
            ok_out = torch.ones(B, dtype=torch.bool) if o == "lm" else (reach_out[:, sid[o]] & reach_in[:, sid[o]])
            keep[:, EIDX[did[d], :n]] = ok_out.unsqueeze(1) & reach_in[:, :n] & reach_out[:, :n]
    return masks * keep


def sample_masks(B):
    prob = torch.sigmoid(gate).unsqueeze(0).expand(B, -1)
    unif = torch.rand_like(prob)
    win = (prob * (1 - prob)).detach()
    prob = win * prob + (1 - win) * prob.detach()
    masks = ((prob - unif) / win + 0.5).clamp(0, 1)
    masks = prune_dangling(masks)
    n_samples = ((masks < 1 - 1e-3) & (masks > 1e-3)).sum(0).float()
    gw = torch.where(n_samples < 1, torch.zeros_like(n_samples), B / n_samples.clamp(min=1))
    return gw * masks + (1 - gw) * masks.detach()


def binary_masks(B):
    return prune_dangling((gate > 0).float().unsqueeze(0).expand(B, -1).clone())


# ------------------------------------------------------------------ pruned forward
def gather(ACT, dests, masks):
    """ACT: (B, n, T, D) the first n sources; returns (B, k, T, D) gated inputs of the k destinations."""
    n = ACT.shape[1]
    idx = torch.stack([EIDX[did[d], :n] for d in dests])                 # (k, n)
    c = masks[:, idx]                                                      # (B, k, n)
    hard = (c < 1e-3).float()
    x = torch.einsum("bkn,bntd->bktd", c, ACT)
    oa = oa_in[:n]
    x = x + torch.einsum("bkn,nd->bkd", (1 - c) * hard, oa).unsqueeze(2) + torch.einsum("bkn,nd->bkd", (1 - c) * (1 - hard), oa.detach()).unsqueeze(2)
    return x + oa_out[[did[d] for d in dests]].view(1, len(dests), 1, D)


QKV = {l: [(l, a, h) for a in ("q", "k", "v") for h in range(H)] for l in range(L)}


def pruned_forward(toks, masks):
    B = toks.shape[0]
    acts = [m.E[toks], m.P[:T].unsqueeze(0).expand(B, -1, -1)]
    for l, ly in enumerate(m.layers):
        ACT = torch.stack(acts, 1)
        X = norm(gather(ACT, QKV[l], masks), QKV[l]).view(B, 3, H, T, D)
        q = torch.einsum("bhtd,hed->bhte", X[:, 0], ly["Wq"]); k = torch.einsum("bhtd,hed->bhte", X[:, 1], ly["Wk"])
        v = torch.einsum("bhtd,hed->bhte", X[:, 2], ly["Wv"])
        s = (q @ k.transpose(-1, -2)) * m.scale
        s = s.masked_fill(~m.mask, float("-inf"))
        z = torch.softmax(s, -1) @ v                                          # (B,H,T,hd)
        heads = torch.einsum("bhte,hed->bhtd", z, ly["Wo"])                   # (B,H,T,D)
        acts += list(heads.unbind(1))
        ACT = torch.stack(acts, 1)
        f = norm(gather(ACT, [(l, "mlp")], masks), [(l, "mlp")])[:, 0]
        acts.append(torch.relu(f @ ly["W1"].T + ly["b1"]) @ ly["W2"].T + ly["b2"])
    ACT = torch.stack(acts, 1)
    x = norm(gather(ACT, [("lm",)], masks), [("lm",)])[:, 0]
    return x[:, -m.n:] @ m.Wlm.T


def kl(lg, tgt):
    return F.kl_div(lg.log_softmax(-1), tgt.log_softmax(-1), log_target=True, reduction="none").sum(-1).mean()


# ------------------------------------------------------------------ training (patching.py, stage 0)
# warm-up (as the original code does before its later stages): gates fixed open, fit the LN constants and the
# destination constants to the original model
PRE = int(sys.argv[8]) if len(sys.argv) > 8 else 150
if PRE > 0:
    opt_p = torch.optim.Adam(([{"params": [logvar], "lr": 0.05}] if lnmode == "lin" else []) + [{"params": [oa_out], "lr": 0.002}])
    ones = torch.ones(UNIQ * REP, E)
    for step in range(PRE):
        w, p = sample_words(m, UNIQ * REP, rng); toks = tokens(m, w, p)
        with torch.no_grad():
            tgt = forward(m, toks)
        loss = kl(pruned_forward(toks, ones), tgt)
        opt_p.zero_grad(); loss.backward(); opt_p.step()
        if (step + 1) % 50 == 0:
            print(f"warm-up {step+1} kl {loss.item():.4f}", flush=True)
opt_g = torch.optim.AdamW([gate], lr=0.1, weight_decay=0, betas=(0.9, 0.995))
groups = [{"params": [oa_in], "lr": 0.002}, {"params": [oa_out], "lr": 1e-4}]
if lnmode == "lin":
    groups.append({"params": [logvar], "lr": 0.01})                # the original code uses 0.1, unstable here
opt_o = torch.optim.AdamW(groups, weight_decay=0)
t0 = time.time(); log = []
countdown = STEPS
for step in range(4 * STEPS):
    w, p = sample_words(m, UNIQ, rng)
    toks = tokens(m, w, p).repeat(REP, 1)
    with torch.no_grad():
        tgt = forward(m, toks)
    masks = sample_masks(toks.shape[0])
    lg = pruned_forward(toks, masks)
    loss_kl = kl(lg, tgt)
    pen = torch.sigmoid(gate).sum()
    loss = loss_kl + lamb * pen
    opt_g.zero_grad(); opt_o.zero_grad(); loss.backward(); opt_g.step(); opt_o.step()
    log.append(loss_kl.item())
    if (step + 1) % 50 == 0:
        g = gate.detach()
        undecided = int(((g > -1) & (g < 1)).sum())
        print(f"step {step+1} kl {np.mean(log[-50:]):.4f} on {int((g > 0).sum())}/{E} undecided {undecided} "
              f"{time.time()-t0:.0f}s", flush=True)
        if undecided > 0:
            countdown = STEPS + 1
        if g.max() < -2:
            print("all pruned"); break
    countdown -= 1
    if countdown <= 0 and step + 1 >= STEPS:
        break

# ------------------------------------------------------------------ evaluation
with torch.no_grad():
    we, pe = sample_words(m, 1000, np.random.default_rng(12345)); te = tokens(m, we, pe)
    match_w, match_s, kls, acc = [], [], [], []
    for i in range(0, 1000, 50):
        tk = te[i:i + 50]
        ref = forward(m, tk); bm = binary_masks(tk.shape[0]); lg = pruned_forward(tk, bm)
        a, b = ref.argmax(-1), lg.argmax(-1)
        match_w.append((a == b).all(1).float()); match_s.append((a == b).float().mean(1)); kls.append(kl(lg, ref).item())
        acc.append(((b - m.num_trans) == pe[i:i + 50]).all(1).float())
    bm = binary_masks(1)[0]
kept = [EDGES[i] for i in range(E) if bm[i] > 0]
res = dict(model=m.spec["name"], lamb=lamb, ln=lnmode, steps=step + 1, seconds=round(time.time() - t0),
           match_word=float(torch.cat(match_w).mean()), match_slot=float(torch.cat(match_s).mean()), kl=float(np.mean(kls)),
           task_word_acc=float(torch.cat(acc).mean()), edges_total=E, edges_kept=len(kept),
           kept=[[list(d), s] for d, s in kept])
print({k: v for k, v in res.items() if k != "kept"})
json.dump(res, open(OUT, "w"), indent=1)
torch.save(dict(gate=gate.detach(), oa_in=oa_in.detach(), oa_out=oa_out.detach(), logvar=logvar.detach()), OUT.replace(".json", ".pt"))
