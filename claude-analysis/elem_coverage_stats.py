"""
Coverage for the elementary model: hops and widths of every slot's chain in the paper's test words and in its training
words (boosted partitioned-window data).

Usage: python elem_coverage_stats.py TEST_CSV TRAIN_CSV [N_TRAIN_WORDS] -> elem_coverage_stats.json
"""
import sys, json
import numpy as np
from elem_tools import trajectories_elem

test_csv, train_csv = sys.argv[1], sys.argv[2]
NTR = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
out = {}
for tag, f, nmax in (("test", test_csv, None), ("train", train_csv, NTR)):
    W = np.loadtxt(f, delimiter=",", dtype=np.int64, max_rows=nmax)
    tr = trajectories_elem(W); h, w = tr["hops"], tr["width"]
    q = lambda a: [float(x) for x in np.quantile(a, [.5, .95, .99, .999])]
    out[tag] = dict(words=int(len(W)), slots=int(h.size), hops_quantiles_50_95_99_999=q(h), hops_max=int(h.max()),
                    width_quantiles_50_95_99_999=q(w), width_max=int(w.max()),
                    frac_width_ge_12=float((w >= 12).mean()), frac_width_ge_11=float((w >= 11).mean()),
                    width_hist=np.bincount(w.ravel(), minlength=17)[1:17].tolist())
    print(tag, {k: v for k, v in out[tag].items() if k != "width_hist"})
json.dump(out, open("elem_coverage_stats.json", "w"), indent=1)
