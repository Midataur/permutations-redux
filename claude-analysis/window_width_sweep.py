"""
Naive window vs partitioned windows: accuracy as the words' letters spread over a wider range.

Words of the model's length whose letters are uniform over a contiguous range of `span` tokens (s_lo..s_lo+span-1,
random offset; the identity token 0 is never used, as in the shifted training words). span = the training window gives
the training distribution; span = n-1 is close to the test distribution (which also has identity letters).
For each span: accuracy at 6 random slots per word, overall and for slots whose chain has at most 3 hops (chains
that short occur in every window word, so their failure cannot be a lack of coverage of the chain).

Usage: python window_width_sweep.py SNAPDIR DATADIR N_WORDS -> window_width_sweep.json
"""
import sys, json
import numpy as np

src = open(__file__.replace("window_width_sweep.py", "window_failure.py")).read().split("cfg = {")[0]
SNAPDIR, DATADIR, NWW = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.argv = ["window_failure.py", SNAPDIR, DATADIR, "1"]
exec(src)          # defines load, evaluate, hops_of, pick, rng

cfg = {"elem-big-1": (25, 20, [10, 12, 15, 19, 24]), "elem-big-2": (25, 20, [10, 12, 15, 19, 24]),
       "elem-wacky-2": (25, 20, [10, 12, 15, 19, 24]), "elem-partitioned-1": (16, 16, [9, 11, 13, 15])}
out = {}
for name, (G, T, spans) in cfg.items():
    m = load(name, G, T); out[name] = {}
    for span in spans:
        lo = rng.integers(1, G - span + 1, size=NWW)
        W = lo[:, None] + rng.integers(0, span, size=(NWW, T))
        H = hops_of(W, G); S = pick(np.ones_like(H, bool), 6)
        _, C = evaluate(m, W, G, S)
        short = S & (H >= 1) & (H <= 3)
        out[name][span] = dict(acc=float(C[S].mean()), acc_short_moved=float(C[short].mean()), n_short=int(short.sum()),
                               acc_unmoved=float(C[S & (H == 0)].mean()) if (S & (H == 0)).sum() else None)
        d = out[name][span]
        print(f"{name:20s} span {span:2d}: acc {d['acc']:.3f}   slots with 1-3 hops {d['acc_short_moved']:.3f} ({d['n_short']})   unmoved {d['acc_unmoved']}")
        sys.stdout.flush()
json.dump(out, open("window_width_sweep.json", "w"), indent=1)
