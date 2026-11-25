import numpy as np
from math import sqrt

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)

R = np.load("strat_results.npz")
ok, hops, dist, moved = R["tf_ok"], R["hops"], R["distinct"], R["moved"]
free_ok = R["free_ok"]
W, n = ok.shape
print(f"words={W:,}  positions={ok.size:,}")
lo, hi = wilson(int(free_ok.sum()), len(free_ok))
print(f"free-running (repo metric, force_valid): {free_ok.sum()}/{len(free_ok)} = {free_ok.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
lo, hi = wilson(int(ok.all(1).sum()), W)
print(f"teacher-forced word accuracy: {ok.all(1).sum()}/{W} = {ok.all(1).mean():.5f}  95% CI [{lo:.5f},{hi:.5f}]")
lo, hi = wilson(int(ok.sum()), ok.size)
print(f"teacher-forced position accuracy: {ok.sum()}/{ok.size} = {ok.mean():.6f}  95% CI [{lo:.6f},{hi:.6f}]")

print("\nBy distinct points visited by the position's trajectory:")
print(f"{'d':>3} {'positions':>10} {'errors':>7}  acc   95% CI")
for d in range(1, dist.max()+1):
    m = dist == d
    k, N = int(ok[m].sum()), int(m.sum())
    if N == 0: continue
    lo, hi = wilson(k, N)
    print(f"{d:>3} {N:>10} {N-k:>7}  {k/N:.5f} [{lo:.5f},{hi:.5f}]")
print("\nBy hops:")
for h in range(0, hops.max()+1):
    m = hops == h
    k, N = int(ok[m].sum()), int(m.sum())
    if N == 0: continue
    lo, hi = wilson(k, N)
    print(f"{h:>3} {N:>10} {N-k:>7}  {k/N:.5f} [{lo:.5f},{hi:.5f}]")

bad = np.argwhere(~ok)
print(f"\nerroneous positions: {len(bad)}")
for w, k in bad:
    print(f"  word {w} pos {k}: hops={hops[w,k]} distinct={dist[w,k]} moved_points={moved[w]} pred={R['tf_pred'][w,k]} true={R['perms'][w,k]}")
print("\nmoved points per word: min/mean/max", moved.min(), moved.mean().round(2), moved.max(), "; share of words moving >10 points:", (moved>10).mean().round(4))
print("word-level max distinct: share >=8:", (dist.max(1)>=8).mean().round(5), " >=9:", (dist.max(1)>=9).mean().round(5), " >=10:", (dist.max(1)>=10).mean().round(5), " >=11:", (dist.max(1)>=11).mean().round(6))
