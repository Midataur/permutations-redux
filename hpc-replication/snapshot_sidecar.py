"""Copy the model file every time training.py rewrites it (once per epoch, after validation).

training.py overwrites model/<MODELNAME>/model.safetensors at the end of every epoch.  This watcher keeps a numbered
copy of each version without touching the training code:  snapshots/epoch_001.safetensors holds the weights after
epoch 1, and so on.  It waits until the file has stopped changing before copying, and checks afterwards that it did
not change during the copy.

Usage: python snapshot_sidecar.py MODEL_FILE OUT_DIR [POLL_SECONDS]
Stops when the file STOP exists in OUT_DIR (the job script creates it after training).
"""
import os, sys, time, shutil

src, out = sys.argv[1], sys.argv[2]
poll = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "sidecar.log"), "a")


def stat():
    try:
        s = os.stat(src)
        return (s.st_mtime_ns, s.st_size)
    except FileNotFoundError:
        return None


last_copied = None
k = 0
while True:
    stopping = os.path.exists(os.path.join(out, "STOP"))
    s1 = stat()
    if s1 is not None and s1 != last_copied:
        time.sleep(3)                         # let the writer finish
        s2 = stat()
        if s2 == s1:
            tmp = os.path.join(out, ".tmp.safetensors")
            shutil.copy2(src, tmp)
            if stat() == s1:                  # unchanged during the copy
                k += 1
                dst = os.path.join(out, f"epoch_{k:03d}.safetensors")
                os.replace(tmp, dst)
                last_copied = s1
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} epoch {k} size {s1[1]}\n"); log.flush()
            continue                          # changed while copying: try again at once
    if stopping:
        break
    time.sleep(poll)
log.write("sidecar stopped\n")
