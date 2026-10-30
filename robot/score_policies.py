"""Score policy rollouts against the public mocap reference with the kit's own Track-3 metric code.

For each public run: the official evaluator's object trajectory export
(fc_out/<name>/eval/object_trajectories_<step>.parquet, read with the kit's read_trajectory) against the released
reference (data/track_3/public, read with the kit's read_episode: visible frames only, slot = observation.objects
order). Rows are matched on frame_index (evaluator: reference_frame = step + start_frame). Then exactly the kit's
pipeline (metric_code/track_3/AUC.py functions): one rigid warp per world on slot 0's first pose, ADD-AUC 1-9 cm,
SP-SR, MP-SR, RPE, MPPE. ADD fixed points: 2000 seeded mesh vertices (the kit's samples are withheld), as in
stages/eval_objects.py. Reference frames the rollout does not cover are reported (not silently dropped).
Usage: python score_policies.py --runs hop_p00_contact:public_ep000000 ... --out scores.json   (flash_chord image)
"""
import os
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import types

ROOT = Path("/scratch/kghasemz/v2d")
KIT = ROOT / "submission_kit/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb.submission_parquet import read_trajectory  # noqa: E402
from v2dlb.trajectory import episode_parquet, read_episode  # noqa: E402

# the kit's metric functions are pure numpy; pandas is only used by its Kaggle score() wrapper (never called here),
# and neither image has it -> exec the source without the pandas imports
_src = (KIT / "metric_code/track_3/AUC.py").read_text()
_src = "\n".join(l for l in _src.splitlines() if not l.startswith(("import pandas", "from pandas")))
M = types.ModuleType("kit_auc")
M.pd = types.SimpleNamespace(DataFrame=object)  # only for the score() signature annotations
exec(compile(_src, "AUC.py", "exec"), M.__dict__)


USE_REF = False
STEPS_PER_FRAME = 0.0


def add_chunked(ach, refp, V, chunk=32):
    """M._object_add over world chunks (the full einsum is T x W x V x 3 per body: tens of GB at 512 worlds)."""
    return np.concatenate([M._object_add(ach[:, w:w + chunk], np.broadcast_to(refp, (ach.shape[0], min(chunk, ach.shape[1] - w)) + refp.shape[2:]), V)
                           for w in range(0, ach.shape[1], chunk)], axis=1)


def verts(name):
    mesh = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj", force="mesh")
    v = np.asarray(mesh.vertices)
    return v[np.random.default_rng(0).choice(len(v), min(2000, len(v)), False)]


def score(fc_name, run):
    O = ROOT / "robot/fc_out" / fc_name / "eval"
    traj_p = sorted(O.glob("object_trajectories_*.parquet"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
    ep = int(run.split("_ep")[-1])
    link = ROOT / "robot/score_links" / fc_name / f"episode_{ep:06d}.parquet"  # the kit reader takes the episode from the name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(traj_p)
    tr = read_trajectory(link)
    ref = read_episode(episode_parquet(ROOT / "data/track_3/public", ep), ep)
    worlds = np.unique(tr.world_index)
    slots = np.unique(tr.object_slot)
    if len(slots) != len(ref.object_names):
        raise ValueError(f"{fc_name}: {len(slots)} rollout slots vs {len(ref.object_names)} reference objects")
    if tr.object_name is not None:
        got = [str(tr.object_name[(tr.object_slot == s)][0]) for s in slots]
        if got != list(ref.object_names):
            print(f"  NOTE {fc_name}: rollout body names {got} vs reference {ref.object_names}")
    if STEPS_PER_FRAME:  # rollout rows are control steps (motion_speed < 1 / control fps != source fps): map each
        # mocap frame f to the step whose playback time is f / src_fps (diagnostic only; not the packer's convention)
        step_of = np.rint(ref.frame_index * STEPS_PER_FRAME).astype(np.int64)
        okf = np.isin(step_of, tr.frame_index)
        remap = dict(zip(step_of[okf], ref.frame_index[okf]))
        selr = np.isin(tr.frame_index, step_of[okf])
        tr.frame_index[selr] = [remap[s] for s in tr.frame_index[selr]]
        tr.frame_index[~selr] = -1
    frames = np.intersect1d(ref.frame_index, np.unique(tr.frame_index))
    missing = len(ref.frame_index) - len(frames)
    sel = np.isin(tr.frame_index, frames)
    o = np.lexsort((tr.object_slot[sel], tr.world_index[sel], tr.frame_index[sel]))
    src = tr.reference_xyzw if USE_REF else tr.pose_xyzw  # --reference: the tracked reference motion itself
    ach = src[sel][o].reshape(len(frames), len(worlds), len(slots), 7)
    keep = np.isin(ref.frame_index, frames)
    refp = ref.pose_xyzw[keep][:, None]  # (T, 1, B, 7)
    ach = M._normalise_quaternions(ach, "rollout", ValueError)
    refp = M._normalise_quaternions(refp, "reference", ValueError)
    ach = M._warp_to_reference_start(ach, refp)
    add = np.concatenate([add_chunked(ach[:, :, [b]], refp[:, :, [b]], verts(n)[None])
                          for b, n in enumerate(ref.object_names)], axis=2)
    auc = M._add_auc(add)
    per = [M._add_auc(add[..., [b]]) for b in range(add.shape[2])]
    return {"episode": ep, "objects": list(ref.object_names), "trajectory": str(traj_p),
            "worlds": int(len(worlds)), "frames_scored": int(len(frames)),
            "ref_frames": int(len(ref.frame_index)), "ref_frames_missing_in_rollout": int(missing),
            "auc": auc, "auc_per_object": dict(zip(ref.object_names, map(float, per))),
            "sp_sr": float(np.mean(M._spider_success(ach, refp))),
            "mp_sr": float(np.mean(M._maniptrans_success(ach, refp, slots))),
            "rpe_cm": M._relative_position_error_cm(ach, refp), "mppe_cm": M._mean_per_point_error_cm(ach, refp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="fc_out_name:run_name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps_per_frame", type=float, default=0.0,
                    help="control steps per mocap frame = control_fps / (source_fps * motion_speed); 0 = rows are frames")
    ap.add_argument("--reference", action="store_true", help="score the reference the policy tracked (sanity check)")
    a = ap.parse_args()
    global USE_REF, STEPS_PER_FRAME
    USE_REF = a.reference
    STEPS_PER_FRAME = a.steps_per_frame
    res = {}
    for s in a.runs:
        fc, run = s.split(":")
        try:
            r = score(fc, run)
            res[fc] = r
            print(f"{fc:28s} AUC {r['auc']:.3f} SP-SR {r['sp_sr']:.2f} MP-SR {r['mp_sr']:.2f} RPE {r['rpe_cm']:.1f} "
                  f"MPPE {r['mppe_cm']:.1f}  frames {r['frames_scored']}/{r['ref_frames']}", flush=True)
        except Exception as e:  # noqa: BLE001
            res[fc] = {"error": repr(e)}
            print(f"{fc:28s} ERROR {e!r}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
