"""
Test 16 for the elementary models: training slots against fresh slots with the same number of hops.

Most elementary runs end with a validation loss 40 to 3,000 times their training loss (snapshot-survey, section 4), as
large a gap as torn-bigger-13's, yet they have no hop-count cliff. This asks whether they too are better on the long
chains of their own training words than on fresh ones, as torn-bigger-13 is (Test 16), and whether it costs them.

Slots are binned by hop count (40-49, 50-59, 60-69, 70-79, 80 or more), at most CAP per bin and source:
  * training: the first NFILES files of the boosted training data (commit fdefb8c, 400,000 words each);
  * validation: the run's own validation file (held out, same generator);
  * fresh: words from the generator of hpc-experiment/exp/datagen.py (matched to the real data), by rejection.
Teacher-forced accuracy and median margin at the slot, per model.

Usage (needs torch for the generator):
  python elem_memorisation.py GIT_REPO NFILES CAP name=path:blocks ...
Writes elem_memorisation_results.json and elem_memorisation_log.txt.
"""
import os, sys, json, time, subprocess
import numpy as np
import pandas as pd
import np_transformer
from np_transformer import NPTransformer
from elem_tools import G, N, trajectories_elem, tf_eval_elem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc-experiment", "exp"))
import torch
from datagen import ElemSampler

_orig = np_transformer.load_safetensors
np_transformer.load_safetensors = lambda p: ({k.replace("_orig_mod.", ""): v for k, v in _orig(p)[0].items()}, None)
repo, NFILES, CAP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
specs = [a.split("=", 1) for a in sys.argv[4:]]
BINS = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 121)]
LOG = open("elem_memorisation_log.txt", "w"); T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time() - T0:5.0f}s] {s}", flush=True); LOG.write(s + "\n"); LOG.flush()


def git_csv(name):
    out = subprocess.run(["git", "-C", repo, "show", f"fdefb8c:scaling-generator/data/elem_partitioned_long_boosted/{name}"],
                         capture_output=True, check=True).stdout
    import io
    return pd.read_csv(io.BytesIO(out), header=None, dtype=np.int8).values.astype(np.int64)


def collect(words_iter, cap):
    """words_iter yields word arrays; keep at most cap slots per hop bin (word, slot, hops, width)."""
    keep = {b: [] for b in BINS}; n = 0
    for W in words_iter:
        n += len(W)
        for s in range(0, len(W), 100_000):
            Wc = W[s:s + 100_000]
            tr = trajectories_elem(Wc, check=False)
            H, Wd = tr["hops"], tr["width"]
            for lo, hi in BINS:
                if len(keep[(lo, hi)]) >= cap:
                    continue
                wi, si = np.nonzero((H >= lo) & (H < hi))
                for a, b in zip(wi, si):
                    if len(keep[(lo, hi)]) < cap:
                        keep[(lo, hi)].append((Wc[a].copy(), int(b), int(H[a, b]), int(Wd[a, b])))
        if all(len(v) >= cap for v in keep.values()):
            break
    return keep, n


def train_iter():
    for i in range(1, NFILES + 1):
        yield git_csv(f"train_data{i}.csv")


def fresh_iter(limit=12_000_000):
    smp = ElemSampler("cpu", 31337)
    made = 0
    while made < limit:
        w = smp.words(200_000).numpy().astype(np.int64); made += len(w); yield w


src = {}
src["training"], nt = collect(train_iter(), CAP); log(f"training: {nt:,} words scanned;", {f"{lo}-{hi - 1}": len(v) for (lo, hi), v in src["training"].items()})
src["validation"], nv = collect([git_csv("val_data.csv")], CAP); log(f"validation: {nv:,} words;", {f"{lo}-{hi - 1}": len(v) for (lo, hi), v in src["validation"].items()})
src["fresh"], nf = collect(fresh_iter(), CAP); log(f"fresh: {nf:,} words generated;", {f"{lo}-{hi - 1}": len(v) for (lo, hi), v in src["fresh"].items()})

res = dict(nfiles=NFILES, cap=CAP, scanned=dict(training=nt, validation=nv, fresh=nf), models={})
for name, rest in specs:
    path, blocks = rest.rsplit(":", 1)
    m = NPTransformer(path, max_group=G, input_len=N, num_trans=G, n_blocks=int(blocks))
    r = {}
    for sname, keep in src.items():
        r[sname] = {}
        for (lo, hi), items in keep.items():
            if not items:
                continue
            W = np.stack([x[0] for x in items]); S = np.array([x[1] for x in items])
            _, c, mg = tf_eval_elem(m, W)
            idx = np.arange(len(W))
            r[sname][f"{lo}-{hi - 1}"] = dict(n=len(W), acc=float(c[idx, S].mean()), margin_median=float(np.median(mg[idx, S])),
                                             margin_p10=float(np.percentile(mg[idx, S], 10)),
                                             width_mean=float(np.mean([x[3] for x in items])))
    res["models"][name] = r
    lines = [f"{name}:"]
    for b in [f"{lo}-{hi - 1}" for lo, hi in BINS]:
        cells = [f"{s} {r[s][b]['acc']:.3f} (n {r[s][b]['n']}, margin {r[s][b]['margin_median']:.1f}, p10 {r[s][b]['margin_p10']:.1f})"
                 for s in r if b in r[s]]
        lines.append(f"   hops {b:6s} " + " | ".join(cells))
    log("\n".join(lines))
    json.dump(res, open("elem_memorisation_results.json", "w"), indent=1)
log("DONE")
