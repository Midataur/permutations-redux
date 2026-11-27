"""Figure for the naive-window section: (a) accuracy as the letters spread over a wider range of tokens;
(b) accuracy at the window's moved slots when j far letters are inserted."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sw = json.load(open("../window_width_sweep.json")); wf = json.load(open("../window_failure.json"))
col = {"elem-big-1": "#c2571a", "elem-big-2": "#e3925c", "elem-wacky-2": "#1f5fa8", "elem-partitioned-1": "#3a8a3a"}
lab = {"elem-big-1": "naive window, run 1 ($S_{25}$)", "elem-big-2": "naive window, run 2 ($S_{25}$)",
       "elem-wacky-2": "two windows of 5 ($S_{25}$)", "elem-partitioned-1": "partitioned windows ($S_{16}$)"}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axs[0]
for m in col:
    if m not in sw:
        continue
    n = 25 if m != "elem-partitioned-1" else 16
    spans = sorted(int(s) for s in sw[m])
    ax.plot(spans, [sw[m][str(s)]["acc_short_moved"] for s in spans], "-o", ms=4, lw=2, color=col[m], label=lab[m])
ax.set_xlabel("range of letters in the word (tokens)"); ax.set_ylabel("accuracy, slots with 1–3 hops")
ax.set_ylim(0.5, 1.02); ax.grid(axis="y", color="#dddddd", lw=.6); ax.set_axisbelow(True)
ax.legend(fontsize=7.5, frameon=False, loc="lower left"); ax.set_title("(a) letters spread over a wider range", fontsize=10)
ax = axs[1]
for m in col:
    if m not in wf:
        continue
    js = sorted(int(j) for j in wf[m]["far letters"])
    ax.plot(js, [wf[m]["far letters"][str(j)]["window_moved_slot_acc"] for j in js], "-o", ms=4, lw=2, color=col[m], label=lab[m])
ax.set_xlabel("letters inserted far from the window, $j$"); ax.set_ylabel("accuracy, moved slots in the window")
ax.set_ylim(0.5, 1.02); ax.set_xticks([0, 2, 8]); ax.grid(axis="y", color="#dddddd", lw=.6); ax.set_axisbelow(True)
ax.set_title("(b) letters that cannot touch the chain", fontsize=10)
for a in axs:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("fig_window.png", dpi=200)
