#!/usr/bin/env python
"""Dump the logged history of your wandb runs to one zip that can be uploaded.

    python export_wandb.py ENTITY/PROJECT [RUN_ID ...]          e.g.  python export_wandb.py mpetschack/scaling-generator torn-bigger-13

Needs `wandb login` on the machine where you run it (your laptop is fine).  For every run it writes the per-epoch history
(training_loss, loss, training_accuracy, validation_accuracy, ...) as CSV, plus the run config and summary as JSON.
The runs of this repo use the run id = MODELNAME (e.g. torn-bigger-13, elem-partitioned-long-boosted-1).
"""
import csv, json, sys, zipfile
import wandb

path = sys.argv[1]
only = set(sys.argv[2:])
api = wandb.Api()
with zipfile.ZipFile("wandb-export.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for run in api.runs(path):
        if only and run.id not in only and run.name not in only:
            continue
        rows = list(run.scan_history())
        keys = sorted({k for r in rows for k in r})
        tmp = f"_{run.id}.csv"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        z.write(tmp, f"{run.id}/history.csv")
        z.writestr(f"{run.id}/meta.json", json.dumps(dict(id=run.id, name=run.name, state=run.state, created_at=str(run.created_at),
                                                          config=dict(run.config), summary={k: v for k, v in run.summary.items() if not k.startswith("_")}),
                                                     indent=1, default=str))
        print(run.id, len(rows), "rows")
print("wrote wandb-export.zip")
