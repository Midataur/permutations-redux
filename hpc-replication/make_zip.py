"""Pack a replication run into one zip to upload.

Contents: the training log without the progress-bar lines, train_exit.txt, the slurm output, env/ (config used,
package versions, accelerate config, GPU list, data checksums), the offline wandb run, and the weight snapshots
after epochs 1-12, every 4th epoch after that, and the last one (every snapshot stays in RUN/snapshots).

Usage: python make_zip.py RUN_DIR ZIP_NAME [KEEP_ALL=0|1]
"""
import os, re, sys, glob, zipfile

run, name = sys.argv[1], sys.argv[2]
keep_all = len(sys.argv) > 3 and sys.argv[3] == "1"
os.chdir(run)

# training log without tqdm lines
with open("train_raw.log", errors="replace") as f, open("train.log", "w") as g:
    for line in f.read().replace("\r", "\n").split("\n"):
        if line.strip() and not re.search(r"\d+%\|", line):
            g.write(line + "\n")

snaps = sorted(glob.glob("snapshots/epoch_*.safetensors"))
keep = []
for i, s in enumerate(snaps, start=1):
    if keep_all or i <= 12 or i % 4 == 0 or i == len(snaps):
        keep.append(s)

files = ["train.log", "train_exit.txt", "job_id.txt", "snapshots/sidecar.log"] + glob.glob("slurm-*.out")
files += [p for p in glob.glob("env/*")] + keep
for root, _, fs in os.walk("wandb"):
    files += [os.path.join(root, f) for f in fs]
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        if os.path.isfile(p):
            z.write(p)
print(f"wrote {name}: {len(keep)} of {len(snaps)} snapshots, {os.path.getsize(name) / 1e6:.0f} MB")
