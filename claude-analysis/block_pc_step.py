"""
Is the block boundary one 'which side' direction in the position embeddings?  For each elementary model, the principal
components of its 120 word-position embeddings (centred); for each of the first 5 components, how well a step at the
model's boundary fits it (R^2 of a two-level step at b) compared with the best step anywhere and with a straight line.
Usage: python block_pc_step.py name=path:b ...  -> block_pc_step.json
"""
import sys, json
import numpy as np
from np_transformer import load_safetensors
res = {}
for name, rest in (a.split("=", 1) for a in sys.argv[1:]):
    path, b = rest.rsplit(":", 1); b = int(b)
    T, _ = load_safetensors(path)
    P = [v for k, v in T.items() if k.endswith("position_embedding.weight")][0].astype(float)[:120]
    X = P - P.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    t = np.arange(120)
    def r2_step(y, c):
        f = np.where(t < c, y[:c].mean(), y[c:].mean()); return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    def r2_line(y):
        A = np.vstack([t, np.ones(120)]).T; f = A @ np.linalg.lstsq(A, y, rcond=None)[0]; return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    comps = []
    for i in range(5):
        y = U[:, i] * S[i]
        steps = {c: r2_step(y, c) for c in range(10, 111)}
        cb = max(steps, key=steps.get)
        comps.append(dict(pc=i + 1, var=round(float(var[i]), 3), step_at_b=round(float(steps[b]), 3), best_step_at=cb,
                          best_step=round(float(steps[cb]), 3), line=round(float(r2_line(y)), 3)))
    res[name] = dict(boundary=b, components=comps)
    print(f"{name:26s} b={b:3d}  " + "  ".join(f"PC{c['pc']} ({100*c['var']:.0f}%): step@b {c['step_at_b']:.2f}, best step {c['best_step']:.2f}@{c['best_step_at']}, line {c['line']:.2f}" for c in comps[:3]))
json.dump(res, open("block_pc_step.json", "w"), indent=1)
