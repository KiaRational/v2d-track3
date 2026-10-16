"""Print a training curve from an offline wandb run (no network): step and the reward/return-like keys."""
import glob, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
import json
f = sorted(glob.glob(sys.argv[1] + "/wandb/offline-run-*/run-*.wandb"))[-1]
ds = datastore.DataStore(); ds.open_for_scan(f)
rows = []
while True:
    d = ds.scan_data()
    if d is None: break
    r = pb.Record(); r.ParseFromString(d)
    if r.WhichOneof("record_type") == "history":
        rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item})
keys = sorted({k for r in rows for k in r})
print(len(rows), "history rows; keys:", [k for k in keys if any(s in k.lower() for s in ("reward", "return", "success", "episode_len", "length", "voc", "termination"))][:40])
want = [k for k in keys if any(s in k.lower() for s in ("return", "reward/total", "episode_length", "success", "voc_scale"))][:8]
step_k = next((k for k in ("environment_steps", "train/environment_steps", "_step") if k in keys), "_step")
for r in rows[:: max(1, len(rows) // 25)] + rows[-1:]:
    print(r.get(step_k), {k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in want if k in r})
