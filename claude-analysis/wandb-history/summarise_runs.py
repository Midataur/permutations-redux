"""One row per run of the full wandb export (knot-theory/scaling-generator, exported 28 Nov 2026)."""
import json, sys, math, csv
ROOT = sys.argv[1]
R = json.load(open(f"{ROOT}/runs.json"))
keys = ["learning_rate", "batch_size", "weight_decay", "n_blocks", "n_embed", "n_head", "dataset_size", "max_group_size",
        "actual_group_size", "max_input_length", "context_length", "dropout", "scheduling", "lr_patience"]
rows = []
for r in R:
    n = r["name"]
    hist = [json.loads(l.replace("NaN", "null").replace("Infinity", "1e999")) for l in open(f"{ROOT}/runs/{r['id']}/history.jsonl")]
    tr = [h.get("training_accuracy") for h in hist]; va = [h.get("validation_accuracy") for h in hist]
    first = lambda xs, th: next((i for i, x in enumerate(xs) if isinstance(x, (int, float)) and x >= th), None)
    last = lambda k: next((h[k] for h in reversed(hist) if isinstance(h.get(k), (int, float))), None)
    c = r["config"]
    row = dict(name=n, family=n.rsplit("-", 1)[0] if n[-1].isdigit() or n.endswith(".0") else n, state=r["state"], created=r["created_at"][:10],
               epochs=len(hist), hours=round((r["summary"].get("_runtime") or 0) / 3600, 1),
               train_acc=last("training_accuracy"), val_acc=last("validation_accuracy"), train_loss=last("training_loss"), val_loss=last("loss"),
               epoch_train90=first(tr, 0.9), epoch_val90=first(va, 0.9), epoch_val99=first(va, 0.99))
    row.update({k: c.get(k) for k in keys})
    rows.append(row)
with open("runs_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "runs")
