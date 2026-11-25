"""Minimal safetensors reader (no torch needed). Converts bfloat16/float16/float32 -> float32 numpy."""
import json, struct
import numpy as np

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        buf = f.read()
    out = {}
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        raw = buf[s:e]
        dt = meta["dtype"]
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"unsupported dtype {dt} for {k}")
        out[k] = arr.reshape(meta["shape"])
    return out, {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
