import os, json, struct, re, sys, datetime
R = "/home/claude/snapshots-extracted/model-snapshots"
rows = []
for name in sorted(os.listdir(R)):
    p = os.path.join(R, name)
    f = os.path.join(p, "model.safetensors") if os.path.isdir(p) else p
    if not os.path.exists(f) or name == "README.md":
        rows.append(dict(name=name, note="no model file")); continue
    d = dict(name=name, size_mb=round(os.path.getsize(f) / 1e6, 1),
             mtime=datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M"))
    if f.endswith(".safetensors"):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; H = json.loads(fh.read(n))
        H.pop("__metadata__", None)
        sh = {k: v["shape"] for k, v in H.items()}; dt = {v["dtype"] for v in H.values()}
        d["dtype"] = ",".join(sorted(dt))
    else:
        d["note"] = "pth"; rows.append(d); continue
    blocks = {int(m.group(1)) for k in sh for m in [re.match(r"blocks\.(\d+)\.sa", k)] if m}
    heads = {int(m.group(1)) for k in sh for m in [re.match(r"blocks\.0\.sa\.heads\.(\d+)\.", k)] if m}
    te = sh.get("token_embedding_table.weight"); pe = sh.get("position_embedding.weight")
    d.update(blocks=len(blocks), heads=len(heads), vocab=te[0] if te else None, D=te[1] if te else None,
             ctx=pe[0] if pe else None, nkeys=len(sh),
             probe=any("probe" in k for k in sh))
    rows.append(d)
json.dump(rows, open("/home/claude/snapwork/inventory.json", "w"), indent=1)
for d in rows:
    print(f"{d['name']:34s} {d.get('mtime',''):16s} {d.get('size_mb',''):>7} blocks={d.get('blocks')} heads={d.get('heads')} vocab={d.get('vocab')} D={d.get('D')} ctx={d.get('ctx')} {d.get('dtype','')} {d.get('note','')}")
