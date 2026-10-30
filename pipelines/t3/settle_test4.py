"""Passive object stability: snap to reference frame f, hold the hands at frame f (zero residual), step T s with
VOC off, report each object's drift. ROBOT=0 disables robot-object/support contact (objects + table only)."""
import os, sys
import numpy as np
from hydra import compose, initialize_config_dir
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene

seq, frames, secs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], float(sys.argv[3])
over = sys.argv[4:]
P = f"{os.environ['HMD']}/ego_recon/processed/sequence_id={seq}/robot_name=sharpa_wave"
with initialize_config_dir(config_dir="/workspace/src/flash_chord/configs", version_base="1.3"):
    cfg = compose("replay", overrides=[f"task.parquet='{P}'", "replay.voc_scale=0.0"] + over)
emb = instantiate_typed(cfg.embodiment, Embodiment)
col = instantiate_typed(cfg.collision, CollisionPolicy)
rc = instantiate_typed(cfg.replay, ReplayConfig)
s = setup_scene(parquet=cfg.task.parquet, control_fps=rc.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=emb,
                collision=col, world_count=1, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
                object_scale_min=1.0, object_scale_max=1.0, object_scale_seed=0, contact_friction=cfg.scene.get("contact_friction", 1.0))
r = ReplayRunner(s.scene, s.reference, rc, device=cfg.device)
ids = r.command.body_ids_w.numpy()
names = getattr(s.scene, "object_body_names", None) or [str(i) for i in ids]
nsteps = int(secs * rc.sim.fps)
def quat_ang(a, b):  # xyzw
    return np.degrees(2 * np.arccos(np.clip(abs((a * b).sum(-1)), 0, 1)))
print(f"{seq} overrides={over} support={cfg.scene.support} ground={col.ground} robot_obj={col.robot_object_collision}")
LIFT = float(os.environ.get("LIFT", "0"))
ALONE = os.environ.get("ALONE", "0") == "1"
for f in frames:
    r.reset(f)
    if LIFT or ALONE:
        qq = r.state_0.body_q.numpy()
        qq[ids, 2] += LIFT
        if ALONE:   # every object but the first far away (2 m to the side, 1 m up): the table and one object only
            for k_, b_ in enumerate(ids[1:]):
                qq[b_, 0] += 2.0 + k_
                qq[b_, 2] += 1.0
        r.state_0.body_q.assign(qq)
        if hasattr(r.state_0, "body_qd"):
            r.state_0.body_qd.zero_()
    q0 = r.state_0.body_q.numpy()[ids].copy()
    traj = []
    for k in range(nsteps):
        r.step(f)
        traj.append(r.state_0.body_q.numpy()[ids].copy())
    traj = np.array(traj)
    for b in range(len(ids)):
        d = np.linalg.norm(traj[:, b, :3] - q0[b, :3], axis=1)
        sp = np.linalg.norm(np.diff(np.r_[q0[None, b, :3], traj[:, b, :3]], axis=0), axis=1) * rc.sim.fps
        rs = [quat_ang(traj[min(int(s * rc.sim.fps), nsteps - 1), b, 3:], q0[b, 3:]) for s in (0.5, 1.5, 3.0, 6.0)]
        print(f"  ROT_T body {b}: " + " ".join(f"{x:.1f}" for x in rs))
        print(f"  f={f:4d} body {names[b] if b < len(names) else b}: z0={q0[b,2]:.4f} drift@0.1s={d[min(4,nsteps-1)]*100:.2f}cm "
              f"@end={d[-1]*100:.2f}cm max={d.max()*100:.2f}cm maxspeed={sp.max():.2f}m/s "
              f"rot@end={quat_ang(traj[-1,b,3:],q0[b,3:]):.1f}deg dz@end={(traj[-1,b,2]-q0[b,2])*100:.2f}cm", flush=True)
