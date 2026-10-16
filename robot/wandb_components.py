"""Per-term reward means + episode length/termination from an offline wandb run at ~10 points in training."""
import glob, json, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
f = sorted(glob.glob(sys.argv[1] + "/wandb/offline-run-*/run-*.wandb"))[-1]
ds = datastore.DataStore(); ds.open_for_scan(f); rows = []
while (d := ds.scan_data()) is not None:
    r = pb.Record(); r.ParseFromString(d)
    if r.WhichOneof("record_type") == "history":
        rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item})
keys = sorted({k for r in rows for k in r})
comp = [k for k in keys if k.startswith("reward_components/") and k.endswith("weighted_mean_per_env_step")]
extra = [k for k in keys if any(s in k for s in ("episode/length_mean", "episode/return_mean", "termination", "curriculum/voc"))][:8]
print("extra keys:", extra)
for r in rows[:: max(1, len(rows) // 8)] + rows[-1:]:
    print("step", r.get("_step"), {k.split("/")[-1][:24]: round(r[k], 3) for k in extra if isinstance(r.get(k), (int, float))})
    print("   ", {k.split("/")[1].replace("_weighted_mean_per_env_step", ""): round(r[k], 3) for k in comp if isinstance(r.get(k), (int, float))})
