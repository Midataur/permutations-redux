"""LLNA check: replace every LayerNorm by a linear map with the variance fixed to its mean over the test distribution."""
import sys, json, numpy as np, torch
from dmodel import *

def forward_lin(m, toks, var):
    x = m.E[toks] + m.P[: toks.shape[1]]
    for i, ly in enumerate(m.layers):
        a = linear_layer_norm(x, ly["ln1w"], ly["ln1b"], var[("ln1", i)])
        out = 0
        for h in range(m.H):
            out = out + attend(m, a @ ly["Wq"][h].T, a @ ly["Wk"][h].T, a @ ly["Wv"][h].T) @ ly["Wo"][h]
        x = x + out + ly["bo"]
        f = linear_layer_norm(x, ly["ln2w"], ly["ln2b"], var[("ln2", i)])
        x = x + torch.relu(f @ ly["W1"].T + ly["b1"]) @ ly["W2"].T + ly["b2"]
    x = linear_layer_norm(x, m.lnfw, m.lnfb, var[("lnf",)])
    return x[:, -m.n:] @ m.Wlm.T

torch.set_grad_enabled(False)
out = {}
for spec in (GENERAL, ELEM):
    m = Weights(spec); rng = np.random.default_rng(1)
    rec = {}
    for _ in range(10):
        w, p = sample_words(m, 64, rng); forward(m, tokens(m, w, p), record=rec)
    var = {k: float(np.mean(v)) for k, v in rec.items()}
    # spread of the per-position variance, to see how far from constant it is
    w, p = sample_words(m, 200, rng); tk = tokens(m, w, p)
    full = forward(m, tk); lin = forward_lin(m, tk, var)
    a, b = full.argmax(-1), lin.argmax(-1)
    r = dict(slot_match=float((a == b).float().mean()), word_match=float((a == b).all(1).float().mean()),
             lin_task_acc=float((b - m.num_trans == p).float().mean()), var={str(k): v for k, v in var.items()})
    out[spec["name"]] = r
    print(spec["name"], {k: v for k, v in r.items() if k != "var"}); print("  LN input variance:", {k: round(v, 2) for k, v in var.items()})
json.dump(out, open("llna_check.json", "w"), indent=1)
