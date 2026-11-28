"""
Replay ReduceLROnPlateau(mode=min, factor=0.1, patience=10, threshold=0.01 relative, eps=1e-8), stepped once per epoch on the
logged training loss (training.py, Nov 2024 - Feb 2025), for every general-transposition run; with k scheduler steps per
epoch for k = 1 and k = 4 (accelerate's prepared scheduler may step once per process).  For each run: the epochs of the
learning-rate cuts, and how much training (optimiser steps, and steps weighted by the learning rate) came before the
learning rate fell below 1% of its starting value, and before the run left the plateau.
"""
import json, sys
ROOT = sys.argv[1]
runs = json.load(open("plateau_escape.json"))


def replay(losses, lr0, k):
    lr, best, bad, lrs = lr0, float("inf"), 0, []
    for x in losses:
        lrs.append(lr)                                     # lr used during this epoch
        for _ in range(k):
            if x is None:
                continue
            if x < best * (1 - 0.01):
                best, bad = x, 0
            else:
                bad += 1
            if bad > 10:
                new = lr * 0.1
                if lr - new > 1e-8:
                    lr = new
                bad = 0
    return lrs


out = []
for d in runs:
    hist = [json.loads(l.replace("NaN", "null").replace("Infinity", "1e999")) for l in open(f"{ROOT}/runs/{d['name']}/history.jsonl")]
    tl = [h.get("training_loss") for h in hist]
    steps_per_epoch = d["words"] / 1024
    row = dict(d)
    for k in (1, 4):
        lrs = replay(tl, d["lr"], k)
        cuts = [i for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]
        low = next((i for i, v in enumerate(lrs) if v < 0.01 * d["lr"]), len(lrs))
        esc = d["escaped_epoch"]
        row[f"k{k}_cut_epochs"] = cuts
        row[f"k{k}_epochs_before_lr_below_1pct"] = low
        row[f"k{k}_steps_before_lr_below_1pct"] = round(low * steps_per_epoch)
        row[f"k{k}_lr_at_escape"] = None if esc is None else lrs[esc]
        row[f"k{k}_lr_weighted_steps_before_escape_or_end"] = sum(lrs[: (esc + 1 if esc is not None else len(lrs))]) * steps_per_epoch
    out.append(row)
    print(f"{d['name']:15s} n={d['group']} ctx={d['context']} words={d['words']/1e6:4.2f}M lr0={d['lr']:.0e} wd={d['wd']:<5} "
          f"escaped={'ep '+str(d['escaped_epoch']) if d['escaped_epoch'] is not None else 'no':7s} | k=1 cuts {row['k1_cut_epochs'][:5]} "
          f"steps before lr<1%: {row['k1_steps_before_lr_below_1pct']:>7} lr at escape {row['k1_lr_at_escape']} | k=4 cuts {row['k4_cut_epochs'][:5]} "
          f"steps before lr<1%: {row['k4_steps_before_lr_below_1pct']:>7} lr at escape {row['k4_lr_at_escape']}")
json.dump(out, open("lr_replay.json", "w"), indent=1)
