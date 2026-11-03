"""Extract training curves from offline wandb runs -> report/curves/<run>.json (downsampled to <= 400 points).
x = environment steps, interpolated from the run's checkpoints (SAC: policy_<env_steps>; PPO: model_<iter> with the
environment_steps stored in the checkpoint metadata)."""
import glob, json, re, struct, sys
import numpy as np
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
KEYS = ["episode/return_mean_per_completed_episode", "reward/total_mean_per_env_step", "reward/dense_mean_per_env_step",
        "episode/reference_progress_mean_per_completed_episode", "episode/length_mean_per_completed_episode",
        "tracking_error/object_position_m_mean_env_max_per_step", "tracking_error/object_orientation_rad_mean_env_max_per_step",
        "reward_components/object_keypoints_weighted_mean_per_env_step", "reward_components/contact_location_weighted_mean_per_env_step",
        "reward_components/termination_weighted_mean_per_env_step", "reward_components/task_completion_weighted_mean_per_env_step",
        "reward_components/reference_progress_weighted_mean_per_env_step"]
def meta(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n)).get("__metadata__", {})
for run in sys.argv[1:]:
    d = f"/scratch/kghasemz/v2d/robot/fc_out/{run}"
    fs = sorted(glob.glob(f"{d}/wandb/offline-run-*/run-*.wandb"))
    if not fs:
        print(run, "no wandb"); continue
    ds = datastore.DataStore(); ds.open_for_scan(fs[-1]); rows = []
    while True:
        x = ds.scan_data()
        if x is None: break
        r = pb.Record(); r.ParseFromString(x)
        if r.WhichOneof("record_type") == "history":
            rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item})
    ck = glob.glob(f"{d}/policy_*.safetensors") or glob.glob(f"{d}/model_*.safetensors")
    total = None
    if ck:
        last = max(ck, key=lambda s: int(re.findall(r"_(\d+)\.safetensors", s)[0]))
        m = meta(last); total = m.get("user.environment_steps")
        total = int(total) if total is not None else (int(re.findall(r"policy_(\d+)", last)[0]) if "policy_" in last else None)
    n = len(rows); idx = np.unique(np.linspace(0, n - 1, min(400, n)).astype(int))
    out = {"run": run, "algo": "ppo" if "_ppo" in run else "sac", "rows": n, "total_env_steps": total,
           "x_frac": (idx / max(n - 1, 1)).tolist(), "series": {}}
    for k in KEYS:
        v = [rows[i].get(k) for i in idx]
        out["series"][k] = [float(a) if isinstance(a, (int, float)) else None for a in v]
    json.dump(out, open(f"/scratch/kghasemz/v2d/report/curves/{run}.json", "w"))
    print(run, n, "rows, env steps", total, flush=True)
