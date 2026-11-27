import subprocess, re, json
G = "/home/claude/midataur/permutations"
log = [l.split(" ", 3) for l in open("/home/claude/snapwork/gitlog.txt").read().splitlines()]
KEYS = ["set_seed", "MAX_GROUP_SIZE", "ACTUAL_GROUP_SIZE", "WINDOW", "WINDOW_COUNT", "PARTITIONED_WINDOWS", "RELABEL",
        "DATA", "MODELNAME", "MAX_TRANS_NUMBER", "INPUT_TYPE", "n_embed", "n_head", "n_blocks", "learning_rate",
        "BATCHSIZE", "weight_decay", "lr_patience", "threshold", "REVERSE_PROBLEM", "num_epochs", "dropout"]
out = {}
for h, d, t, subj in log:
    try:
        cfg = subprocess.run(["git", "-C", G, "show", f"{h}:scaling-generator/config.py"], capture_output=True, text=True).stdout
    except Exception:
        continue
    if not cfg:
        continue
    c = {}
    for k in KEYS:
        m = re.search(r"^set_seed\((\d+)\)" if k == "set_seed" else rf"^{k}\s*=\s*(.+?)\s*(#.*)?$", cfg, re.M)
        if m:
            c[k] = m.group(1).strip()
    out[h] = dict(date=f"{d} {t}", subject=subj, cfg=c)
json.dump(out, open("/home/claude/snapwork/configs.json", "w"), indent=1)
for h, v in out.items():
    c = v["cfg"]
    if re.search(r"rerun|boosted|elem-part|reverse", v["subject"]):
        print(f"{h} {v['date']} {v['subject'][:32]:32s} name={c.get('MODELNAME')} seed={c.get('set_seed')} data={c.get('DATA')} G={c.get('MAX_GROUP_SIZE')}/{c.get('ACTUAL_GROUP_SIZE')} N={c.get('MAX_TRANS_NUMBER')} win={c.get('WINDOW')},{c.get('PARTITIONED_WINDOWS')} blocks={c.get('n_blocks')} bs={c.get('BATCHSIZE')} lr={c.get('learning_rate')} rev={c.get('REVERSE_PROBLEM')}")
