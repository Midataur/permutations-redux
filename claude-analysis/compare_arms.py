"""
Put the per-arm results of analyze_run.py side by side:  python compare_arms.py OUT_DIR  ->  OUT_DIR/summary.md (also printed)
Per arm: planted accuracy by hop count (and beyond the 10-point support), the probe staircase at the word positions (the deepest
remaining-hop bin that a layer's probe decodes at 50% or more and well above layer 0), the number of heads whose zeroing costs more than 10 points, the
dominant hop offsets per block, and the three costliest heads at the longest L.
"""
import glob, json, os, sys
import numpy as np

out = sys.argv[1]
arms = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)) and os.path.exists(os.path.join(out, d, "run_info.json")))
order = ["g_m10", "g_plant10", "g_plant10_b8", "e_base"]
arms = sorted(arms, key=lambda a: order.index(a) if a in order else 99)
lines = []
P = lambda s="": lines.append(s)


def first(pattern, d):
    g = sorted(glob.glob(os.path.join(d, pattern)))
    return json.load(open(g[0])) if g else None


for arm in arms:
    d = os.path.join(out, arm)
    info = json.load(open(os.path.join(d, "run_info.json")))
    kind = info["kind"]
    P(f"## {arm}  ({kind}, {info['blocks']} blocks, checkpoint step {info['step']})\n")
    acc = first("arm_accuracy_*_results.json", d)
    if acc:
        P("Planted-chain accuracy (teacher-forced, at the planted slot):\n")
        P("| series | " + ("L | s" if kind == "general" else "L | width") + " | correct | % | median margin |")
        P("|---|---|---|---|---|---|")
        for c in acc["cells"]:
            P(f"| {c['series']} | {c['L']} | {c.get('s', c.get('w'))} | {c['ok']}/{c['n']} | {100*c['ok']/c['n']:.1f} | {c['margin_median']:.1f} |")
        P()
    pr = first("input_probe*.json", d)
    if pr:
        bl = pr["by_layer_and_remaining_hops"]
        bins = list(bl["0"].keys())
        P("Probe at the word positions (accuracy of the continuation probe by remaining hops; chance 4% general / 6% elementary):\n")
        P("| layer | " + " | ".join(bins) + " |")
        P("|---|" + "---|" * len(bins))
        for l in sorted(bl, key=int):
            P(f"| {l} | " + " | ".join(f"{100*bl[l][b][0]:.0f}" for b in bins) + " |")
        reach = []
        for l in sorted(bl, key=int):
            ok = [b for b in bins if l in bl and bl[l][b][1] > 0 and bl[l][b][0] >= max(0.5, bl["0"][b][0] + 0.15)]
            reach.append(f"{l}: up to h={ok[-1].split('-')[1]}" if ok else f"{l}: none")
        P("\nDeepest bin decoded at 50% or more and at least 15 points above layer 0, by layer: " + "; ".join(reach) + "\n")
    ab = first("head_ablation_[ge]*_results.json", d)
    at = first("attention_[ge]*_results.json", d)
    if ab:
        P("Heads whose zeroing at the word positions costs more than 10 points (of the 'word' ablation, blocks 1 to n-1; block n is never read):\n")
        P("| L | model right on planted | costly heads | three costliest (block, head: accuracy) |")
        P("|---|---|---|---|")
        for L in sorted(ab["by_L"], key=int):
            r = ab["by_L"][L]
            a = np.array(r["acc_word"])[:-1]
            idx = np.argsort(a, axis=None)[:3]
            top = ", ".join(f"b{i // a.shape[1] + 1}h{i % a.shape[1]}: {a.flat[i]:.2f}" for i in idx)
            P(f"| {L} | {100*r['model_right_on_planted']:.0f}% | {int((a < 0.9).sum())} of {a.size} | {top} |")
        P()
    if at:
        Lmax = sorted(at["by_L"], key=int)[-1]
        r = at["by_L"][Lmax]
        O = np.array(r["offset_mass"]) * at["N"]; offs = np.array(r["offsets"])
        P(f"Strongest hop offsets per head at L={Lmax} (head, offset d: times uniform attention; only heads with a peak above 8x):\n")
        for l in range(O.shape[0]):
            items = []
            for h in range(O.shape[1]):
                j = np.argsort(-O[l, h])[:2]
                if O[l, h, j[0]] > 8 and offs[j[0]] != 0:
                    items.append(f"h{h}: {offs[j[0]]:+d} ({O[l, h, j[0]]:.0f}x)" + (f", {offs[j[1]]:+d} ({O[l, h, j[1]]:.0f}x)" if O[l, h, j[1]] > 8 else ""))
            P(f"- block {l + 1}: " + ("; ".join(items) if items else "none"))
        P()
    gr = first("greedy_ablation_*results.json", d)
    if gr:
        P(f"Greedy head ablation (zero, at each step, the one further word-position head that hurts most; L={gr['L']}, {gr['n_words']} words that the intact model gets right):\n")
        P(", ".join(f"+{x['added']} -> {x['acc']:.2f}" for x in gr["steps"]))
        P()
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
