"""Simulator-settled initial rest poses: let the simulator decide how each object lies on the table.

Our resting object poses carry a ~10-15 deg tilt error (mostly about the camera's horizontal axis: the least observable
direction from a head-mounted camera), and export_result_bundle's rest settle refuses tilts > 8 deg or judges
stability from the visual mesh / its own COM rather than the collision pieces and URDF inertia the simulator uses. So
at reset every resting object tips 11-15 deg (passive test, beat1/diag/settle) -- the robot sees objects fall over.

Per sequence (in the flash_chord image): reset to frame 0 with no robot contacts, step SETTLE_S seconds, read every
object body's settled pose. In the sequence parquet, each body's INITIAL REST (frames before it moves > MOVE_M or turns
> TURN_DEG from frame 0 in the reference) is replaced by the settled pose; the last BLEND frames before it starts
moving ease back to the reference (slerp / lerp). Writes the parquet in place (backup <file>.presettle, never
compounded) and a report. Usage (cwd = flash_chord source):
  HMD=<tree> python sim_settle_rest.py <seq> [<seq> ...]"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from scipy.spatial.transform import Rotation, Slerp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene

SETTLE_S, MOVE_M, TURN_DEG, BLEND = 2.0, 0.005, 3.0, 5


def settle(parquet):
    over = [f"task.parquet='{parquet}'", "replay.voc_scale=0.0", "collision.ground=false",
            "collision.robot_object_collision=false", "collision.robot_support_collision=false"]
    with initialize_config_dir(config_dir="/workspace/src/flash_chord/configs", version_base="1.3"):
        cfg = compose("replay", overrides=over)
    emb = instantiate_typed(cfg.embodiment, Embodiment)
    col = instantiate_typed(cfg.collision, CollisionPolicy)
    rc = instantiate_typed(cfg.replay, ReplayConfig)
    s = setup_scene(parquet=cfg.task.parquet, control_fps=rc.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=emb,
                    collision=col, world_count=1, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
                    object_scale_min=1.0, object_scale_max=1.0, object_scale_seed=0,
                    contact_friction=cfg.scene.get("contact_friction", 1.0))
    r = ReplayRunner(s.scene, s.reference, rc, device=cfg.device)
    ids = r.command.body_ids_w.numpy()
    r.reset(0)
    q0 = r.state_0.body_q.numpy()[ids].copy()
    for _ in range(int(SETTLE_S * rc.sim.fps)):
        r.step(0)
    return q0, r.state_0.body_q.numpy()[ids].copy()     # (B, 7): px py pz qx qy qz qw


def main():
    hmd = os.environ["HMD"]
    report = {}
    for seq in sys.argv[1:]:
        d = f"{hmd}/ego_recon/processed/sequence_id={seq}/robot_name=sharpa_wave"
        f = glob.glob(d + "/*.parquet")[0]
        if not os.path.exists(f + ".presettle"):
            shutil.copy(f, f + ".presettle")
        q0, q1 = settle(f + ".presettle")
        t = pq.read_table(f + ".presettle")
        pos = np.array(t.column("object_body_position").to_pylist(), float)     # (T, B, 3)
        wxyz = np.array(t.column("object_body_wxyz").to_pylist(), float)        # (T, B, 4)
        T_, B = pos.shape[:2]
        # the simulator's frame-0 state must be the parquet's frame 0 (else the frames differ: refuse)
        ref0 = np.c_[pos[0], wxyz[0][:, 1:], wxyz[0][:, :1]]
        if np.abs(ref0[:, :3] - q0[:, :3]).max() > 0.01:
            raise RuntimeError(f"{seq}: simulator reset differs from parquet frame 0: {ref0[:, :3]} vs {q0[:, :3]}")
        rep = []
        for b in range(B):
            R_ref = Rotation.from_quat(np.c_[wxyz[:, b, 1:], wxyz[:, b, :1]])
            mv = np.linalg.norm(pos[:, b] - pos[0, b], axis=1) > MOVE_M
            tn = (R_ref * R_ref[0].inv()).magnitude() > np.radians(TURN_DEG)
            k = int(np.argmax(mv | tn)) if (mv | tn).any() else T_
            if k < 2:
                rep.append({"body": b, "rest_frames": k, "applied": False})
                continue
            Rs = Rotation.from_quat(q1[b, 3:])
            ps = q1[b, :3]
            for u in range(k):
                al = 0.0 if u < k - BLEND else (u - (k - BLEND) + 1) / (BLEND + 1)   # ease back to the reference
                R = Slerp([0, 1], Rotation.concatenate([Rs, R_ref[u]]))(al)
                p = (1 - al) * ps + al * pos[u, b]
                q = R.as_quat()
                wxyz[u, b] = [q[3], q[0], q[1], q[2]]
                pos[u, b] = p
            rep.append({"body": b, "rest_frames": k, "applied": True,
                        "tilt_change_deg": round(float(np.degrees((Rs * R_ref[0].inv()).magnitude())), 1),
                        "move_cm": round(float(np.linalg.norm(ps - ref0[b, :3])) * 100, 2)})
        cols = {c: t.column(c) for c in t.column_names}
        cols["object_body_position"] = pa.array(pos.tolist(), type=t.schema.field("object_body_position").type)
        cols["object_body_wxyz"] = pa.array(wxyz.tolist(), type=t.schema.field("object_body_wxyz").type)
        if "object_root_position" in cols:   # root = body 0
            cols["object_root_position"] = pa.array(pos[:, 0].tolist(), type=t.schema.field("object_root_position").type)
        if "object_root_axis_angle" in cols:
            rv = Rotation.from_quat(np.c_[wxyz[:, 0, 1:], wxyz[:, 0, :1]]).as_rotvec()
            cols["object_root_axis_angle"] = pa.array(rv.tolist(), type=t.schema.field("object_root_axis_angle").type)
        out = pa.table(cols, schema=t.schema)
        pq.write_table(out, f)
        report[seq] = rep
        print(seq, json.dumps(rep), flush=True)
    with open(f"{hmd}/sim_settle_rest_report.json", "a") as fh:
        fh.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
