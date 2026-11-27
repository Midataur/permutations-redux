"""
How well trained is each elementary model in the Spartan model folder? (Context for block_survey.py.)

For every elementary model with the 120 + 16 position layout: teacher-forced accuracy on the real test file of
elem-partitioned-long-boosted-1 (uniform words over 16 points; first N_TEST words), and on planted chains: window
width 8 at L = 40 and 80 hops, and width 16 at L = 24 (the width effect of section 3.8).

Usage: python elem_model_survey.py SNAPSHOT_DIR TEST_CSV [N_TEST] [N_PLANT] [EXTRA name=path ...]
Writes elem_model_survey.json.
"""
import os, sys, json, time
import numpy as np
import np_transformer
from np_transformer import NPTransformer
from elem_tools import G, N, tf_eval_elem, plant_elem_batch

_orig = np_transformer.load_safetensors


def _strip(path):
    W, meta = _orig(path)
    return {k.replace("_orig_mod.", ""): v for k, v in W.items()}, meta


np_transformer.load_safetensors = _strip

root, test_csv = sys.argv[1], sys.argv[2]
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 200
extra = dict(a.split("=", 1) for a in sys.argv[5:])
words = np.loadtxt(test_csv, delimiter=",", dtype=np.int64)[:NT]
inv = {d["name"]: d for d in json.load(open("/home/claude/snapwork/inventory.json"))}
cells = [(8, 40), (8, 80), (16, 24)]
out = json.load(open("elem_model_survey.json")) if os.path.exists("elem_model_survey.json") else {}
models = {n: os.path.join(root, n, "model.safetensors") for n, d in inv.items()
          if d.get("ctx") == 136 and d.get("vocab") == 34 and not n.startswith("reverse")}
models["elem-partitioned-long-boosted-2"] = os.path.join(root, "elem-partitioned-long-boosted-2", "model.safetensors")
models.update(extra)
t0 = time.time()
for name, path in sorted(models.items()):
    if name in out:
        continue
    blocks = inv.get(name, {}).get("blocks") or 5
    m = NPTransformer(path, max_group=G, input_len=N, num_trans=G, n_blocks=blocks)
    _, corr, marg = tf_eval_elem(m, words)
    r = dict(blocks=blocks, test_slots=float(corr.mean()), test_words=float(corr.all(1).mean()))
    rng = np.random.default_rng(2611)
    for w, L in cells:
        pw, ks = plant_elem_batch(rng, NP, L, w)
        _, c, _ = tf_eval_elem(m, pw)
        r[f"w{w}_L{L}"] = float(c[np.arange(NP), ks].mean())
    out[name] = r
    json.dump(out, open("elem_model_survey.json", "w"), indent=1)
    print(f"[{time.time() - t0:5.0f}s] {name:34s} blocks {blocks}  test slots {r['test_slots']:.4f} words {r['test_words']:.3f}  "
          + "  ".join(f"w{w} L{L}: {r[f'w{w}_L{L}']:.2f}" for w, L in cells), flush=True)
print("DONE")
