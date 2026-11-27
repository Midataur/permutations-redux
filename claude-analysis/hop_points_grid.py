"""
Where does torn-bigger-13's decline start, and does it follow the number of hops or the number of distinct points?

1. Planted chains on a grid: L = 10 ... 22 hops, s = 3, 5, 7, 10 distinct points (s <= L+1), N words per cell,
   teacher-forced accuracy at the planted slot.
2. Fresh natural slots (natural 10-point words relabelled into 25; the cached set of cliff_survey.py, 300 per hop count
   for h = 11 ... 16), split by the number of distinct points the slot's chase visits.

Models: torn-bigger-13 as in the repo (February 2025), torn-bigger-13 after the November resume, and the retrained
control g_m10 and g_plant10 for comparison.

Usage: python hop_points_grid.py N_PER_CELL name=path:blocks ...   -> hop_points_grid.json
"""
import sys, json, time
import numpy as np
from np_transformer import NPTransformer
from permdata import plant_batch, trajectories
from evalutil import tf_eval

NC = int(sys.argv[1])
specs = [a.split("=", 1) for a in sys.argv[2:]]
z = np.load("/home/claude/snapwork/cliff_fresh.npz")
FW, FS, FH = z["w"].astype(np.int64), z["s"], z["h"]
FD = trajectories(FW, check=False)["distinct"][np.arange(len(FW)), FS]
Ls, Ss = (10, 12, 13, 14, 15, 16, 17, 18, 20, 22), (3, 5, 7, 10)
out, T0 = {}, time.time()
for name, rest in specs:
    path, blocks = rest.rsplit(":", 1)
    m = NPTransformer(path, n_blocks=int(blocks))
    grid = {}
    for L in Ls:
        for s in Ss:
            if s > L + 1:
                continue
            rng = np.random.default_rng(1000 * L + s)
            w, k = plant_batch(rng, NC, L, s)
            _, c, _, _ = tf_eval(m, w, bs=200)
            grid[f"{L},{s}"] = float(c[np.arange(NC), k].mean())
    _, c, _, _ = tf_eval(m, FW, bs=200)
    ok = c[np.arange(len(FW)), FS]
    nat = {}
    for h in range(11, 17):
        for lo, hi in ((2, 6), (7, 8), (9, 10)):
            sel = (FH == h) & (FD >= lo) & (FD <= hi)
            if sel.sum() >= 10:
                nat[f"{h},{lo}-{hi}"] = [int(ok[sel].sum()), int(sel.sum())]
    out[name] = dict(planted=grid, natural=nat)
    json.dump(out, open("hop_points_grid.json", "w"), indent=1)
    print(f"[{time.time() - T0:4.0f}s] {name}", flush=True)
    print("   planted  L\\s " + "".join(f"{s:>7d}" for s in Ss))
    for L in Ls:
        print(f"            {L:3d} " + "".join(f"{grid[f'{L},{s}']:7.2f}" if f"{L},{s}" in grid else "       " for s in Ss))
    print("   natural  h: points 2-6 | 7-8 | 9-10")
    for h in range(11, 17):
        cells = [nat.get(f"{h},{lo}-{hi}") for lo, hi in ((2, 6), (7, 8), (9, 10))]
        print(f"            {h:3d}: " + " | ".join(f"{a / b:.3f} ({b})" if (a, b) != (None, None) and b else "   -    " for a, b in
                                                 [(x[0], x[1]) if x else (None, None) for x in cells]))
