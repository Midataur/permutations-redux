"""n*(N): the smallest training group S_n that covers S_N test words up to an uncovered mass eps (section 3.27)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scaling_law import n_star
Ns = np.unique(np.round(np.logspace(1, 3, 40)).astype(int))
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
cols = {1e-2: "#9cc3e6", 1e-4: "#4a8fd0", 1e-6: "#1f4f8a"}
for ax, (lab, f) in zip(axs, (("test words of $N$ letters ($\\lambda\\approx2$)", lambda N: N),
                              ("test words of $2N$ letters ($\\lambda\\approx4$)", lambda N: 2 * N),
                              ("test words of 25 letters", lambda N: 25))):
    for eps, c in cols.items():
        ax.plot(Ns, [n_star(N, f(N), eps) for N in Ns], "-", color=c, lw=2, label=f"uncovered mass $\\leq 10^{{{int(np.log10(eps))}}}$")
    if f(25) == 25:   # the paper's pair: 25-letter words on 25 points
        ax.plot([25], [10], "o", color="#c2571a", ms=6, zorder=5)
    ax.set_xscale("log"); ax.set_title(lab, fontsize=9); ax.set_xlabel("test group size $N$", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(color="#e5e5e5", lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("smallest training group size $n^*$", fontsize=8)
axs[0].legend(fontsize=7, frameon=False, loc="upper left")
axs[2].annotate("the paper: $S_{10}\\to S_{25}$", (25, 10), xytext=(40, 13), fontsize=7, color="#c2571a", arrowprops=dict(arrowstyle="-", color="#c2571a", lw=.8))
plt.tight_layout(); plt.savefig("fig_scaling_law.png", dpi=200)
