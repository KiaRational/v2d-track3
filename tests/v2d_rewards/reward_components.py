"""Summarize an offline wandb run: every reward_components/* key (mean / first / last / min / max over history
rows), non-finite numeric values, and selected training keys. Usage: python reward_components.py <output_dir>"""
import glob
import json
import math
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

f = sorted(glob.glob(sys.argv[1] + "/wandb/offline-run-*/run-*.wandb"))[-1]
ds = datastore.DataStore()
ds.open_for_scan(f)
rows = []
while True:
    d = ds.scan_data()
    if d is None:
        break
    r = pb.Record()
    r.ParseFromString(d)
    if r.WhichOneof("record_type") == "history":
        rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item})
print(f, len(rows), "history rows")
keys = sorted({k for r in rows for k in r})
bad = sorted({k for r in rows for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)})
print("non-finite keys:", bad or "none")
def series(k):
    return [r[k] for r in rows if isinstance(r.get(k), (int, float))]
for k in keys:
    if k.startswith("reward_components/") or any(s in k for s in ("episode_return", "episode_length", "reference_progress", "voc", "stage", "steps_per", "sps", "termination", "reference_end", "tracking_error")):
        s = series(k)
        if s:
            print(f"{k:70s} mean {sum(s)/len(s):+.5f} first {s[0]:+.5f} last {s[-1]:+.5f} min {min(s):+.5f} max {max(s):+.5f} n {len(s)}")
