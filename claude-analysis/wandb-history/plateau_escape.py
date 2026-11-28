"""Which general-transposition runs left the ~0.72 plateau, against training-set size and training length (wandb export)."""
import csv, json, sys
ROOT = sys.argv[1]
rows = list(csv.DictReader(open("runs_summary.csv")))
f = lambda x: float(x) if x not in ("", None) else None
out = []
for r in rows:
    n = r["name"]
    if not n.startswith(("torn-big", "torn-bigger", "torn-masked", "torn-1", "torn_long")):
        continue
    hist = [json.loads(l.replace("NaN", "null").replace("Infinity", "1e999")) for l in open(f"{ROOT}/runs/{n}/history.jsonl")]
    va = [h.get("validation_accuracy") for h in hist]
    esc = next((i for i, v in enumerate(va) if isinstance(v, (int, float)) and v >= 0.9), None)
    out.append(dict(name=n, created=r["created"], words=int(float(r["dataset_size"])), group=int(float(r["max_group_size"])),
                    context=int(float(r["context_length"])), lr=f(r["learning_rate"]), wd=f(r["weight_decay"]), blocks=int(float(r["n_blocks"])),
                    epochs=len(hist), hours=f(r["hours"]), final_val=f(r["val_acc"]), escaped_epoch=esc,
                    word_views_millions=round(len(hist) * int(float(r["dataset_size"])) / 1e6, 1),
                    word_views_at_escape_millions=None if esc is None else round((esc + 1) * int(float(r["dataset_size"])) / 1e6, 1)))
out.sort(key=lambda d: (d["group"], d["context"], d["created"]))
for d in out:
    print(f"{d['name']:16s} {d['created']} n={d['group']} ctx={d['context']:2d} words={d['words']/1e6:5.2f}M lr={d['lr']:.0e} wd={d['wd']:<6} "
          f"blocks={d['blocks']} epochs={d['epochs']:4d} ({d['hours']:5.1f} h, {d['word_views_millions']:7.1f}M views)  final val {d['final_val']:.3f}  "
          f"escaped at epoch {d['escaped_epoch']} ({d['word_views_at_escape_millions']}M views)")
json.dump(out, open("plateau_escape.json", "w"), indent=1)
