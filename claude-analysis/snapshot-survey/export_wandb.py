"""
Export the wandb runs needed for the snapshot survey into one zip to upload.

For each run: run.json (config, summary, state, dates), history.csv (every logged row, unsampled), and the files
wandb-metadata.json, config.yaml, output.log and requirements.txt when the run has them.

Usage (anywhere you are logged in to wandb, e.g. the Spartan login node):
    python export_wandb.py                      # the default list below
    python export_wandb.py rerun-1 rerun-2      # or just some runs
Writes wandb-export-snapshots.zip in the current folder. Runs it cannot find are listed in not_found.txt inside the zip.
"""
import os, sys, json, shutil
import wandb

ENTITY, PROJECT = "knot-theory", "scaling-generator"
RUNS = [
    # elementary, boosted config: which runs form a position block, and how long each trained
    "elem-partitioned-long-boosted-2", "elem-partitioned-long-boosted-3", "elem-partitioned-long-boosted-4",
    "rerun-1", "rerun-2", "rerun-3", "rerun-4",
    # elementary, older data and architecture
    "elem-partitioned-long-5", "elem-partitioned-long-11", "elem-partitioned-long-12", "elem-partitioned-long-14",
    # general: the runs that stayed on the plateau, and torn-bigger-9 (leaky mask)
    "torn-bigger-5", "torn-bigger-7", "torn-bigger-9", "torn-bigger-10", "torn-bigger-11", "torn-bigger-12",
]
names = sys.argv[1:] or RUNS
api = wandb.Api(timeout=60)
out = "wandb-export-snapshots"
os.makedirs(out, exist_ok=True)
missing = []


def find(name):
    # training.py uses the model name as the run id; fall back to searching by display name
    try:
        return api.run(f"{ENTITY}/{PROJECT}/{name}")
    except Exception:
        pass
    try:
        rs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": name}))
        return rs[0] if rs else None
    except Exception:
        return None


for name in names:
    r = find(name)
    if r is None:
        print(f"not found: {name}")
        missing.append(name)
        continue
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    json.dump({"id": r.id, "name": r.name, "state": r.state, "created": r.created_at, "config": dict(r.config),
               "summary": r.summary._json_dict}, open(os.path.join(d, "run.json"), "w"), indent=1, default=str)
    rows = list(r.scan_history())
    keys = sorted({k for row in rows for k in row})
    with open(os.path.join(d, "history.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join("" if row.get(k) is None else str(row.get(k)) for k in keys) + "\n")
    for f in r.files():
        if f.name in ("wandb-metadata.json", "config.yaml", "output.log", "requirements.txt"):
            try:
                f.download(root=d, replace=True)
            except Exception as e:
                print(f"  {name}: could not download {f.name}: {e}")
    print(f"{name}: {len(rows)} history rows, state {r.state}")

open(os.path.join(out, "not_found.txt"), "w").write("\n".join(missing) + "\n")
shutil.make_archive(out, "zip", out)
print(f"wrote {out}.zip ({len(names) - len(missing)} runs, {len(missing)} not found)")
