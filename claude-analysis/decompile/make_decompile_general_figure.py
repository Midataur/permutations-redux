"""Match accuracy against D-RASP program size for the three general models after stage-1 pruning (section 3.26)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lines import count_lines

MODELS = [("general", "torn-bigger-13", "#1f5fa8", "o"), ("g_m10", "g_m10", "#c2571a", "s"), ("g_plant10", "g_plant10", "#3a8a3a", "^")]
FULL = 165818738
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for key, name, col, mk in MODELS:
    pts = []
    for lam in ("0.001", "0.0003"):
        r = json.load(open(f"runs/{key}_lin_l{lam}.json")); k = [(tuple(d), s) for d, s in r["kept"]]
        pts.append((count_lines(k), r["match_word"], lam))
    f0 = f"runs/{key}_lin_l0_long.json" if key != "general" else "runs/general_lin_l0.json"
    r0 = json.load(open(f0)); pts.append((FULL, r0["match_word"], "0"))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col, lw=2, ms=6, label=name)
    for x, y, lam in pts:
        if lam == "0":
            continue
        ax.annotate(f"λ={lam}", (x, y), textcoords="offset points", xytext=(5, -11), fontsize=6.5, color="#555555")
ax.axhline(0.9, color="#888888", lw=1, ls=":"); ax.text(1.2e7, 0.91, "paper's bar (0.9)", fontsize=7, color="#666666")
ax.set_xscale("log"); ax.set_ylim(0.2, 1.02)
ax.set_xlabel("D-RASP lines, each MLP one line (right-hand points: λ = 0, the full model)", fontsize=8)
ax.set_ylabel("words on which all outputs match the model", fontsize=8)
ax.tick_params(labelsize=7); ax.grid(color="#e5e5e5", lw=.6); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.set_title("Stage-1 pruning of the three general models (per-position linear LayerNorm)", fontsize=9)
plt.tight_layout(); plt.savefig("../fig_decompile_general.png", dpi=200)
