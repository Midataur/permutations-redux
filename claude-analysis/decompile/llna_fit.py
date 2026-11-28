"""
LLNA, fitted: every LayerNorm's variance is replaced by learned constants (log-variance parameters), trained to minimise
the KL divergence to the original model at the output positions, with all components kept (the paper's 'original model
with linear LN', which it requires to reach a match accuracy of 0.9).  Three granularities:
  per_ln       one constant per LayerNorm (the paper's assumption)
  per_type     one constant per LayerNorm and position type (word positions / output positions)
  per_pos      one constant per LayerNorm and position
Match accuracy: share of words whose argmax prediction agrees with the original at every output position
(teacher-forced with the true permutation), on 1,000 fresh test words.
Usage: python llna_fit.py STEPS -> llna_fit.json
"""
import sys, json, time, numpy as np, torch
from dmodel import *

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
GRANS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["per_ln", "per_type", "per_pos"]
torch.manual_seed(0)


def forward_param(m, toks, logv, idx):
    """idx: (T,) long, which constant each position uses; logv: dict name -> (K,) parameter."""
    def ln(x, w, b, name):
        v = logv[name].exp()[idx].view(1, -1, 1)
        return (x - x.mean(-1, keepdim=True)) / torch.sqrt(v + 1e-5) * w + b
    x = m.E[toks] + m.P[: toks.shape[1]]
    for i, ly in enumerate(m.layers):
        a = ln(x, ly["ln1w"], ly["ln1b"], f"ln1.{i}")
        out = 0
        for h in range(m.H):
            out = out + attend(m, a @ ly["Wq"][h].T, a @ ly["Wk"][h].T, a @ ly["Wv"][h].T) @ ly["Wo"][h]
        x = x + out + ly["bo"]
        f = ln(x, ly["ln2w"], ly["ln2b"], f"ln2.{i}")
        x = x + torch.relu(f @ ly["W1"].T + ly["b1"]) @ ly["W2"].T + ly["b2"]
    x = ln(x, m.lnfw, m.lnfb, "lnf")
    return x[:, -m.n:] @ m.Wlm.T


def ln_stats(m, toks):
    """mean LN-input variance per LN and position (T,)"""
    st = {}
    x = m.E[toks] + m.P[: toks.shape[1]]
    for i, ly in enumerate(m.layers):
        st[f"ln1.{i}"] = x.var(-1, unbiased=False).mean(0)
        a = layer_norm(x, ly["ln1w"], ly["ln1b"]); out = 0
        for h in range(m.H):
            out = out + attend(m, a @ ly["Wq"][h].T, a @ ly["Wk"][h].T, a @ ly["Wv"][h].T) @ ly["Wo"][h]
        x = x + out + ly["bo"]
        st[f"ln2.{i}"] = x.var(-1, unbiased=False).mean(0)
        f = layer_norm(x, ly["ln2w"], ly["ln2b"]); x = x + torch.relu(f @ ly["W1"].T + ly["b1"]) @ ly["W2"].T + ly["b2"]
    st["lnf"] = x.var(-1, unbiased=False).mean(0)
    return st


out = {}
for spec in (GENERAL, ELEM):
    m = Weights(spec); rng = np.random.default_rng(3)
    with torch.no_grad():
        w, p = sample_words(m, 256, rng); st = ln_stats(m, tokens(m, w, p))
        we, pe = sample_words(m, 1000, rng); te = tokens(m, we, pe)
        ref = torch.cat([forward(m, te[i:i + 100]) for i in range(0, 1000, 100)])
    T = m.T; res = {}
    for gran in GRANS:
        if gran == "per_ln":
            idx = torch.zeros(T, dtype=torch.long); K = 1
        elif gran == "per_type":
            idx = torch.zeros(T, dtype=torch.long); idx[m.N:] = 1; K = 2
        else:
            idx = torch.arange(T); K = T
        logv = {}
        for k, v in st.items():
            init = torch.stack([v[idx == j].mean() for j in range(K)])
            logv[k] = torch.nn.Parameter(init.log())
        opt = torch.optim.Adam(logv.values(), lr=0.05)
        t0 = time.time()
        for step in range(STEPS):
            w, p = sample_words(m, 32, rng); tk = tokens(m, w, p)
            with torch.no_grad():
                tgt = forward(m, tk)
            lg = forward_param(m, tk, logv, idx)
            loss = torch.nn.functional.kl_div(lg.log_softmax(-1), tgt.log_softmax(-1), log_target=True, reduction="none").sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lg = torch.cat([forward_param(m, te[i:i + 100], logv, idx) for i in range(0, 1000, 100)])
            a, b = ref.argmax(-1), lg.argmax(-1)
            r = dict(word_match=float((a == b).all(1).float().mean()), slot_match=float((a == b).float().mean()),
                     final_train_kl=float(loss), seconds=round(time.time() - t0))
        res[gran] = r
        print(spec["name"], gran, r); sys.stdout.flush()
    out[spec["name"]] = res
json.dump(out, open("llna_fit.json", "w"), indent=1)
