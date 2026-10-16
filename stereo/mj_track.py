"""Physically plausible hand-object data by physics-based tracking in MuJoCo, with videos (venv-mj).

Instead of replaying the reconstructed hands kinematically (infinitely stiff: a finger slightly
inside the object ejects it instead of gripping), each hand is an ARTICULATED MuJoCo hand that
tracks the reconstruction compliantly, and physics produces the data:
  hand     MANO's kinematic tree at this subject's bone lengths (mj_prep.py): wrist body pulled to
           the reconstructed wrist pose by a soft weld, 15 finger bodies x 3 hinges driven by
           position servos toward the reconstructed joint angles, collision capsules on every bone,
           gravity-compensated. Physics stops the fingers at the object and table surfaces -- no
           penetration by construction -- and where the reference presses into the object the
           servo keeps pressing: a bounded, compliant grip force.
  objects  free rigid bodies (CoACD convex parts of the scan, URDF mass/inertia, friction 1.0),
           starting at their reconstructed pose; the table is the measured stereo plane.
  loop     simulate -> diagnose -> correct, per grasp segment: the segment is simulated from the
           saved state at its start; if the held object drifts more than --slip_cm from its
           reconstructed trajectory, grip is raised (servo gain x2, up to --max_level times) and
           the segment is re-simulated.
Outputs:
  stereo/<out>/<side>/NNNNNN.json   the SIMULATED hands in the usual MANO schema (default mano_fit_mj)
  <run>/sim_mj/<obj>/poses/          the simulated object trajectory (camera frame)
  stereo/<out>/report.json          per object error vs reconstruction, per segment grip level,
                                    hand-object penetration from the contact solver
  $V2D_VIZ/10_mujoco/<run>.mp4      [third-person | close-up] renders; green ghost = reconstructed
                                    object pose, so drift is visible directly
Usage: MUJOCO_GL=egl python mj_track.py <run> [--prep mj_prep_mano_fit.json] [--no_video]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
F = np.diag([-1.0, 1.0, 1.0])
CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
PARENT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7, 9: 0, 10: 9, 11: 10, 12: 11,
          13: 0, 14: 13, 15: 14, 16: 15, 17: 0, 18: 17, 19: 18, 20: 19}
ACT_J = [j for c in CHAINS for j in c[:3]]  # 15 actuated joints
OBJ_RGB = ["0.25 0.45 0.9 1", "0.9 0.55 0.25 1", "0.35 0.8 0.45 1"]


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return f"{w} {x} {y} {z}"


def v3(v):
    return " ".join(f"{x:.6f}" for x in v)


def urdf_inertial(name):
    u = ROOT / "data/track_3/public/urdf" / f"{name}_rigid.urdf"
    if not u.exists():  # objects with no provided scan (wooden_spoon): our own URDF, same 0.3 kg placeholder
        u = ROOT / "urdf_extra" / f"{name}_rigid.urdf"
    if not u.exists():
        return None
    x = u.read_text()
    m = float(re.search(r'<mass value="([^"]+)"', x).group(1))
    com = [float(v) for v in re.search(r'<inertial>\s*<origin xyz="([^"]+)"', x).group(1).split()]
    return m, com, [float(re.search(rf'{k}="([^"]+)"', x).group(1)) for k in ("ixx", "iyy", "izz")]


SKIN = "0.87 0.7 0.58 1"


def part_geoms(side, body, parts):
    """MANO mesh piece of this body: convex hull for collision (group 3), real surface for rendering."""
    if str(body) not in parts:
        return []
    return [f'<geom type="mesh" mesh="{side}_hull{body}" contype="2" conaffinity="1" condim="4" mass="0.01" group="3" '
            f'rgba="{SKIN}"/>',
            f'<geom type="mesh" mesh="{side}_vis{body}" contype="0" conaffinity="0" density="0" rgba="{SKIN}"/>']


def hand_assets(side, parts):
    return [f'<mesh name="{side}_{kind}{b}" file="{d[kind]}"/>' for b, d in parts.items() for kind in ("hull", "vis")]


def hand_xml(side, rest, parts=None, r_th=0.0095, r_f=0.008, r_palm=0.011):
    """Nested MuJoCo bodies for one MANO hand in its rest frame (world-aligned axes). With `parts`, every body
    carries its piece of this subject's MANO mesh (collision = convex hull, visual = MANO surface); without,
    the old capsule approximation."""
    x = [f'<body name="{side}_root" pos="0 0 -5" gravcomp="1"><freejoint name="{side}_free"/>']
    if parts:
        x += part_geoms(side, 0, parts)
    else:
        x.append(f'<geom type="sphere" size="0.012" contype="2" conaffinity="1" mass="0.05" rgba="{SKIN}"/>')
        for k in (1, 5, 9, 13, 17):  # palm: wrist -> each finger base, plus the knuckle line
            x.append(f'<geom type="capsule" fromto="0 0 0 {v3(rest[k] - rest[0])}" size="{r_palm}" contype="2" '
                     f'conaffinity="1" mass="0.02" rgba="{SKIN}"/>')
        x.append(f'<geom type="capsule" fromto="{v3(rest[5] - rest[0])} {v3(rest[17] - rest[0])}" size="{r_palm}" '
                 f'contype="2" conaffinity="1" mass="0.02" rgba="{SKIN}"/>')

    def chain(c, i):
        j = c[i]
        if i == 3:  # tip: no body
            return []
        r = r_th if c[0] == 1 else r_f
        out = [f'<body name="{side}_j{j}" pos="{v3(rest[j] - rest[PARENT[j]])}" gravcomp="1">']
        for ax, a in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
            out.append(f'<joint name="{side}_j{j}{ax}" type="hinge" axis="{a}" damping="0.02" armature="0.001"/>')
        if parts:
            out += part_geoms(side, j, parts)
        else:
            out.append(f'<geom type="capsule" fromto="0 0 0 {v3(rest[c[i + 1]] - rest[j])}" size="{r}" contype="2" '
                       f'conaffinity="1" mass="0.01" condim="4" rgba="{SKIN}"/>')
        out += chain(c, i + 1)
        out.append("</body>")
        return out

    for c in CHAINS:
        x += chain(c, 0)
    x.append("</body>")
    x.append(f'<body name="{side}_target" mocap="true" pos="0 0 -5"/>')
    return x


def look_xyaxes(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f"{v3(r)} {v3(u)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prep", default="mj_prep_mano_fit.json")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--out", default="mano_fit_mj")
    ap.add_argument("--sub", type=int, default=20)
    ap.add_argument("--kp", type=float, default=1.0, help="finger servo gain, N m / rad")
    ap.add_argument("--max_level", type=int, default=3, help="grasp-closing escalations allowed per grasp")
    ap.add_argument("--close_step", type=float, default=0.002, help="fingertip squeeze per level (m)")
    ap.add_argument("--slip_cm", type=float, default=1.5)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--weld_tc", type=float, default=0.04,
                    help="wrist spring time constant (s): soft enough that contact deflects the hand instead of the "
                         "reconstructed wrist shoving the object, stiff enough to carry it")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--capsule_hand", action="store_true", help="old capsule hand instead of the MANO mesh hand")
    ap.add_argument("--debug_frames", type=int, nargs="*", default=[])
    ap.add_argument("--pregrasp", type=int, default=10, help="frames before lift-off at which the fingers start closing")
    ap.add_argument("--ik_every", type=int, default=5, help="substeps between fingertip IK updates")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    meta = json.loads((run / "episode.json").read_text())
    fps = meta["fps"]
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    ts = sorted(slam)
    prep = json.loads((run / "stereo" / a.prep).read_text())
    tp = json.loads((run / "table_plane.json").read_text())
    up, table_h = np.array(tp["up"]), float(tp["height"])
    rep = json.loads((run / "fusion_report.json").read_text())

    objs = []
    for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
        pd = run / a.variant / o["name"] / "poses"
        W = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in ts if (pd / f"{t:06d}.json").exists()}
        if W:
            parts = sorted((ROOT / "cache/coacd" / o["name"]).glob("part_*.obj"))
            objs.append({"name": o["name"], "W": W, "parts": parts})
    cen = np.mean([o["W"][min(o["W"])][:3, 3] for o in objs], 0)
    table_c = cen - up * (cen @ up - table_h)
    cam_c = np.mean([slam[t][:3, 3] for t in ts], 0)
    back = cam_c - cen
    back -= up * (back @ up)
    back /= np.linalg.norm(back)

    # ------------------------------------------------------------------ model
    x = [f'<mujoco><option timestep="{1.0 / (fps * a.sub):.6f}" gravity="{v3(-9.81 * up)}" cone="elliptic" '
         f'impratio="10"/><compiler autolimits="true"/>',
         f'<default><geom friction="{a.friction} 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>',
         '<visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual><asset>',
         '<texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.72 0.58" rgb2="0.76 0.66 0.52" width="256" height="256"/>',
         '<material name="table" texture="grid" texrepeat="8 8"/>']
    for o in objs:
        for k, p in enumerate(o["parts"]):
            x.append(f'<mesh name="{o["name"]}_{k}" file="{p}"/>')
        x.append(f'<mesh name="{o["name"]}_vis" file="{ROOT / "meshes_obj" / o["name"] / (o["name"] + ".obj")}"/>')
    for s_ in ("left", "right"):
        if s_ in prep["sides"] and not a.capsule_hand:
            x += hand_assets(s_, prep["sides"][s_].get("mesh_parts", {}))
    x.append("</asset><worldbody>")
    x.append(f'<light pos="{v3(cen + up * 2)}" dir="{v3(-up)}" directional="true" castshadow="false" diffuse="0.7 0.7 0.7"/>')
    x.append(f'<geom name="table" type="box" size="0.8 0.8 0.02" pos="{v3(table_c - up * 0.02)}" '
             f'quat="{quat(Rotation.align_vectors([up], [[0, 0, 1.0]])[0].as_matrix())}" material="table" '
             f'contype="1" conaffinity="3"/>')
    eye3 = cen + Rotation.from_rotvec(np.deg2rad(55) * up).as_matrix() @ back * 0.9 + up * 0.55
    eyec = cen + Rotation.from_rotvec(np.deg2rad(-35) * up).as_matrix() @ back * 0.45 + up * 0.3
    x.append(f'<camera name="third" pos="{v3(eye3)}" xyaxes="{look_xyaxes(eye3, cen, up)}" fovy="50"/>')
    x.append(f'<camera name="close" pos="{v3(eyec)}" xyaxes="{look_xyaxes(eyec, cen, up)}" fovy="50"/>')
    for i, o in enumerate(objs):
        W0 = o["W"][min(o["W"])]
        ine = urdf_inertial(o["name"])
        x.append(f'<body name="{o["name"]}" pos="{v3(W0[:3, 3])}" quat="{quat(W0[:3, :3])}"><freejoint/>')
        if ine:
            x.append(f'<inertial pos="{v3(ine[1])}" mass="{ine[0]}" diaginertia="{v3(ine[2])}"/>')
        for k in range(len(o["parts"])):
            x.append(f'<geom type="mesh" mesh="{o["name"]}_{k}" density="0" contype="1" conaffinity="3" condim="4" group="3"/>')
        x.append(f'<geom type="mesh" mesh="{o["name"]}_vis" contype="0" conaffinity="0" density="0" rgba="{OBJ_RGB[i % 3]}"/>')
        x.append("</body>")
        x.append(f'<body name="{o["name"]}_ghost" mocap="true" pos="{v3(W0[:3, 3])}" quat="{quat(W0[:3, :3])}">'
                 f'<geom type="mesh" mesh="{o["name"]}_vis" contype="0" conaffinity="0" rgba="0.2 0.95 0.3 0.3"/></body>')
    sides = [s for s in ("left", "right") if s in prep["sides"]]
    for s in sides:
        x += hand_xml(s, np.array(prep["sides"][s]["rest_joints"]),
                      None if a.capsule_hand else prep["sides"][s].get("mesh_parts"))
    x.append("</worldbody><equality>")
    for s in sides:
        x.append(f'<weld body1="{s}_target" body2="{s}_root" solref="{a.weld_tc} 1" solimp="0.9 0.95 0.001"/>')
    x.append("</equality><actuator>")
    for s in sides:
        for j in ACT_J:
            for ax in "xyz":
                x.append(f'<position name="{s}_a{j}{ax}" joint="{s}_j{j}{ax}" kp="{a.kp}" forcerange="-2 2"/>')
    x.append("</actuator></mujoco>")
    model = mujoco.MjModel.from_xml_string("\n".join(x))
    data = mujoco.MjData(model)

    obj_bid = {o["name"]: model.body(o["name"]).id for o in objs}
    ghost_mid = {o["name"]: model.body(o["name"] + "_ghost").mocapid[0] for o in objs}
    tgt_mid = {s: model.body(f"{s}_target").mocapid[0] for s in sides}
    act = {s: [model.actuator(f"{s}_a{j}{ax}").id for j in ACT_J for ax in "xyz"] for s in sides}
    root_q = {s: model.jnt_qposadr[model.joint(f"{s}_free").id] for s in sides}
    hinge_q = {s: [model.jnt_qposadr[model.joint(f"{s}_j{j}{ax}").id] for j in ACT_J for ax in "xyz"] for s in sides}
    hand_geoms = {g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.split("_")[0] in sides}
    kp0 = model.actuator_gainprm[:, 0].copy()

    def ref_hand(s, t):
        fr = prep["sides"][s]["frames"].get(str(t))
        if fr is None:
            return None
        return np.array(fr["root_R"]), np.array(fr["root_p"]), np.array([fr["hinge"][str(j)] for j in ACT_J]).ravel()

    closing = {}  # side -> {"obj": o, "tips": {tip joint: (point, outward normal) in object frame}, "squeeze": m}
    TIPJ = [4, 8, 12, 16, 20]
    RAD = {4: 0.0095, 8: 0.008, 12: 0.008, 16: 0.008, 20: 0.008}
    surf = {}
    for o in objs:
        import trimesh
        from scipy.spatial import cKDTree
        m = trimesh.load(ROOT / "meshes_obj" / o["name"] / f"{o['name']}.obj", force="mesh")
        P, fi = trimesh.sample.sample_surface(m, 20000, seed=0)
        surf[o["name"]] = (cKDTree(P), P, m.face_normals[fi])

    def fk_all(s, root_R, root_p, hin):
        rest = np.array(prep["sides"][s]["rest_joints"])
        J = np.zeros((21, 3))
        Rc = {0: root_R}
        J[0] = root_p
        for j in range(1, 21):
            pa = PARENT[j]
            J[j] = J[pa] + Rc[pa] @ (rest[j] - rest[pa])
            Rc[j] = Rc[pa] @ (Rotation.from_euler("XYZ", hin[j]).as_matrix() if j in hin else np.eye(3))
        return J

    def cone_cols(P, N, c, mu=0.5, k=8):
        cols = []
        for p_, n_ in zip(P, N):  # n_: outward normal; contact force pushes inward
            nin = -n_
            a_ = np.array([1.0, 0, 0]) if abs(nin[0]) < 0.9 else np.array([0, 1.0, 0])
            t1 = np.cross(nin, a_)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(nin, t1)
            for th in np.linspace(0, 2 * np.pi, k, endpoint=False):
                e = nin + mu * (np.cos(th) * t1 + np.sin(th) * t2)
                e /= np.linalg.norm(e)
                cols.append(np.r_[e, np.cross(p_ - c, e) / 0.1])
        return np.array(cols).T

    def support_residual(P, N, c, g_obj):
        """How well contacts (object frame) can hold the object against gravity (g_obj: gravity dir, object frame)."""
        from scipy.optimize import nnls
        if len(P) == 0:
            return 1.0
        b = np.r_[-g_obj, np.zeros(3)]  # the hands must supply the negative of gravity, no net torque
        _, r_ = nnls(cone_cols(P, N, c), b, maxiter=500)
        return r_ / np.linalg.norm(b)

    def plan_tips(s, t, o, max_d=0.04, radius=0.03, K=30, lam=3.0):
        """Fingertip contact points (object frame) chosen near where each fingertip was observed so that together
        they can hold the object against gravity (force closure within friction cones), by coordinate descent."""
        tips0 = plan_tips_nearest(s, t, o, max_d)
        if len(tips0) < 2:
            return tips0
        tree, P, N = surf[o["name"]]
        ine = urdf_inertial(o["name"])
        c = np.array(ine[1]) if ine else P.mean(0)
        g_obj = o["W"][t][:3, :3].T @ (-up)
        r = ref_hand(s, t)
        hin = {j: r[2][3 * k:3 * k + 3] for k, j in enumerate(ACT_J)}
        J = fk_all(s, r[0], r[1], hin)
        Wi = np.linalg.inv(o["W"][t])
        cand, obs = {}, {}
        for tj in tips0:
            q = Wi[:3, :3] @ J[tj] + Wi[:3, 3]
            idx = tree.query_ball_point(q, radius)
            idx = idx[:: max(1, len(idx) // K)][:K] if idx else [tree.query(q)[1]]
            cand[tj], obs[tj] = idx, q
        choice = {tj: tree.query(obs[tj])[1] for tj in tips0}

        def cost(ch):
            idx = list(ch.values())
            dist = sum(np.linalg.norm(P[ch[tj]] - obs[tj]) for tj in ch)
            return support_residual(P[idx], N[idx], c, g_obj) + lam * dist / len(ch)
        best = cost(choice)
        for _ in range(3):
            for tj in tips0:
                for k in cand[tj]:
                    trial = dict(choice)
                    trial[tj] = k
                    v = cost(trial)
                    if v < best - 1e-4:
                        best, choice = v, trial
        res = support_residual(P[list(choice.values())], N[list(choice.values())], c, g_obj)
        print(f"      {s} grasp plan on {o['name']} at frame {t}: {len(choice)} fingertips, gravity-support residual "
              f"{res:.2f} (nearest points: "
              f"{support_residual(P[[tree.query(obs[tj])[1] for tj in tips0]], N[[tree.query(obs[tj])[1] for tj in tips0]], c, g_obj):.2f})",
              flush=True)
        return {tj: (P[k], N[k]) for tj, k in choice.items()}

    def plan_tips_nearest(s, t, o, max_d=0.04):
        """Each fingertip within max_d of the object gets its nearest surface point, fixed in the object frame."""
        r = ref_hand(s, t)
        if r is None or t not in o["W"]:
            return {}
        hin = {j: r[2][3 * k:3 * k + 3] for k, j in enumerate(ACT_J)}
        J = fk_all(s, r[0], r[1], hin)
        Wi = np.linalg.inv(o["W"][t])
        tree, P, N = surf[o["name"]]
        out = {}
        for tj in TIPJ:
            q = Wi[:3, :3] @ J[tj] + Wi[:3, 3]
            d, k = tree.query(q)
            if d < max_d:
                out[tj] = (P[k], N[k])
        return out

    def ik_finger(s, root_R, root_p, hin, tj, target):
        """Hinges of the finger ending at tj that bring its tip to target, staying near the reference."""
        from scipy.optimize import least_squares
        chain = [c for c in CHAINS if c[-1] == tj][0][:3]
        x0 = np.concatenate([hin[j] for j in chain])

        def res(x):
            h = dict(hin)
            for k, j in enumerate(chain):
                h[j] = x[3 * k:3 * k + 3]
            return np.r_[(fk_all(s, root_R, root_p, h)[tj] - target) * 100, 0.05 * (x - x0)]
        sol = least_squares(res, x0, max_nfev=30)
        return {j: sol.x[3 * k:3 * k + 3] for k, j in enumerate(chain)}

    def close_hinges(s, hinge, t, root_R, root_p):
        c = closing.get(s)
        if c is None or t not in c["obj"]["W"] or t not in c["frames"]:
            return hinge
        hin = {j: hinge[3 * k:3 * k + 3] for k, j in enumerate(ACT_J)}
        # grasp the object where it actually is in the simulation (it may have been nudged on approach),
        # not where the reconstruction expected it
        bid_ = obj_bid[c["obj"]["name"]]
        W = np.eye(4)
        W[:3, :3], W[:3, 3] = data.xmat[bid_].reshape(3, 3), data.xpos[bid_]
        dbg = t == c.get("plan_frame")
        J0 = fk_all(s, root_R, root_p, hin) if dbg else None
        for tj, (p, n) in c["tips"].items():
            # fingertip capsule end placed so its surface presses `squeeze` into the object at p
            target = W[:3, :3] @ (p + n * (RAD[tj] - c["squeeze"])) + W[:3, 3]
            hin.update(ik_finger(s, root_R, root_p, hin, tj, target))
            if dbg:
                J1 = fk_all(s, root_R, root_p, hin)
                print(f"      IK {s} tip{tj}: ref->target {100 * np.linalg.norm(J0[tj] - target):.1f} cm, "
                      f"after IK {100 * np.linalg.norm(J1[tj] - target):.2f} cm", flush=True)
        out = np.concatenate([hin[j] for j in ACT_J])
        if dbg:
            print(f"      {s} hinge change max {np.abs(out - hinge).max():.3f} rad", flush=True)
        return out

    ik_cache = {}

    def set_targets(t, al, t2):
        for s in sides:
            r1, r2 = ref_hand(s, t), ref_hand(s, t2)
            if r1 is None:  # hand not reconstructed: park it far away
                data.mocap_pos[tgt_mid[s]] = [0, 0, -5]
                continue
            if r2 is None:
                r2 = r1
            R = Rotation.from_matrix([r1[0], r2[0]])
            Rm = Rotation.from_rotvec((1 - al) * R[0].as_rotvec() + al * R[1].as_rotvec()) if al else R[0]
            data.mocap_pos[tgt_mid[s]] = (1 - al) * r1[1] + al * r2[1]
            x_, y_, z_, w_ = Rm.as_quat()
            data.mocap_quat[tgt_mid[s]] = [w_, x_, y_, z_]
            h = (1 - al) * r1[2] + al * r2[2]
            if s in closing:  # IK targets re-aimed every --ik_every substeps from the current wrist and object
                key = (s, t, int(al * a.sub) // a.ik_every)
                if key not in ik_cache:
                    # finger IK from the SIMULATED wrist (the soft wrist spring lets it sit slightly off the
                    # reconstruction; a few degrees there moves fingertips by centimetres)
                    qa = root_q[s]
                    w_, x_, y_, z_ = data.qpos[qa + 3:qa + 7]
                    ik_cache[key] = close_hinges(s, r1[2], t, Rotation.from_quat([x_, y_, z_, w_]).as_matrix(),
                                                 data.qpos[qa:qa + 3].copy())
                h = ik_cache[key]
            data.ctrl[act[s]] = h
        for o in objs:
            W = o["W"].get(t)
            if W is not None:
                data.mocap_pos[ghost_mid[o["name"]]] = W[:3, 3]
                x_, y_, z_, w_ = Rotation.from_matrix(W[:3, :3]).as_quat()
                data.mocap_quat[ghost_mid[o["name"]]] = [w_, x_, y_, z_]

    # start: hands at their first reconstructed pose (collisions resolve any initial overlap gently)
    t0 = ts[0]
    set_targets(t0, 0, t0)
    for s in sides:
        r = ref_hand(s, min(int(k) for k in prep["sides"][s]["frames"]))
        if r is not None:
            qa = root_q[s]
            data.qpos[qa:qa + 3] = r[1]
            x_, y_, z_, w_ = Rotation.from_matrix(r[0]).as_quat()
            data.qpos[qa + 3:qa + 7] = [w_, x_, y_, z_]
            data.qpos[hinge_q[s]] = r[2]
    mujoco.mj_forward(model, data)
    # FK agreement: MuJoCo's own forward kinematics of the reference hinges vs the reconstructed joints
    for s in sides:
        fr = prep["sides"][s]["frames"].get(str(min(int(k) for k in prep["sides"][s]["frames"])))
        rest_ = np.array(prep["sides"][s]["rest_joints"])
        Jref = np.array(fr["joints_world"])
        errs_ = []
        for c in CHAINS:
            for j, nxt in zip(c[:3], c[1:]):
                b_ = model.body(f"{s}_j{j}").id
                pj = data.xpos[b_]
                pn = data.xpos[b_] + data.xmat[b_].reshape(3, 3) @ (rest_[nxt] - rest_[j])
                errs_ += [np.linalg.norm(pj - Jref[j]), np.linalg.norm(pn - Jref[nxt])]
        print(f"{s}: MuJoCo FK vs reconstructed joints: mean {1000 * np.mean(errs_):.1f} mm, max {1000 * np.max(errs_):.1f} mm",
              flush=True)

    # grasp segments (any hand on any object), in time order
    held = sorted(set().union(*[set(v) for d in rep.values() if isinstance(d, dict)
                                for v in d.get("hand_contact", {}).values()] or [set()]))
    segs, cur = [], []
    for t in held:
        if cur and t - cur[-1] > 2:
            segs.append(cur)
            cur = []
        cur.append(t)
    segs += [cur] if cur else []
    seg_of = {t: i for i, sg in enumerate(segs) for t in sg}

    renderer = None
    if not a.no_video:
        renderer = mujoco.Renderer(model, 480, 640)
        vopt = mujoco.MjvOption()
        vopt.geomgroup[3] = 0
    wrist_err = {}
    frames_out, sim_obj, sim_hand, levels, pen = [], {o["name"]: {} for o in objs}, {s: {} for s in sides}, [], 0.0

    geom_side = {g: model.body(model.geom_bodyid[g]).name.split("_")[0] for g in hand_geoms}
    obj_of_body = {obj_bid[o["name"]]: o["name"] for o in objs}
    contact_log = {}
    f6 = np.zeros(6)

    def step_frame(t, t2):
        nonlocal pen
        for k in range(a.sub):
            set_targets(t, k / a.sub, t2)
            mujoco.mj_step(model, data)
        log = {}
        for ci, c in enumerate(data.contact[:data.ncon]):
            if (c.geom1 in hand_geoms) != (c.geom2 in hand_geoms):
                pen = max(pen, -c.dist * 1000)
                hg, og = (c.geom1, c.geom2) if c.geom1 in hand_geoms else (c.geom2, c.geom1)
                nm = obj_of_body.get(model.geom_bodyid[og])
                if nm is None:
                    continue
                mujoco.mj_contactForce(model, data, ci, f6)
                key = f"{geom_side[hg]}>{nm}"
                n_, fn = log.get(key, (0, 0.0))
                log[key] = (n_ + 1, fn + abs(f6[0]))
        contact_log[t] = {k: [v[0], round(v[1], 2)] for k, v in log.items()}

    def obj_errs(t):
        return {o["name"]: float(np.linalg.norm(data.xpos[obj_bid[o["name"]]] - o["W"][t][:3, 3]) * 100)
                for o in objs if t in o["W"]}

    def obj_err(t):
        return max(obj_errs(t).values(), default=0.0)

    def record(t):
        for o in objs:
            M = np.eye(4)
            M[:3, :3] = data.xmat[obj_bid[o["name"]]].reshape(3, 3)
            M[:3, 3] = data.xpos[obj_bid[o["name"]]]
            sim_obj[o["name"]][t] = M
        for s in sides:
            qa = root_q[s]
            r_ = ref_hand(s, t)
            if r_ is not None:
                wrist_err.setdefault(s, {})[t] = round(float(np.linalg.norm(data.qpos[qa:qa + 3] - r_[1]) * 100), 2)
            w_, x_, y_, z_ = data.qpos[qa + 3:qa + 7]
            sim_hand[s][t] = (Rotation.from_quat([x_, y_, z_, w_]).as_matrix(), data.qpos[qa:qa + 3].copy(),
                              data.qpos[hinge_q[s]].copy())
        if renderer is not None:
            imgs = []
            for cam in ("third", "close"):
                renderer.update_scene(data, camera=cam, scene_option=vopt)
                imgs.append(renderer.render())
            frames_out.append((t, np.hstack(imgs)))

    i = 0
    while i < len(ts):
        t = ts[i]
        si = seg_of.get(t)
        if si is None or segs[si][0] != t:
            step_frame(t, ts[min(i + 1, len(ts) - 1)])
            record(t)
            i += 1
            continue
        # a grasp starts: simulate it, raising the grip until the held object follows
        seg = [u for u in segs[si] if u in slam]
        j_end = ts.index(seg[-1])
        sig = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.empty(mujoco.mj_stateSize(model, sig))
        mujoco.mj_getState(model, data, state, sig)
        n_rec = len(frames_out)
        best = None  # (mean error, level)
        tries = list(range(a.max_level + 1))
        k_try = 0
        while True:
            level = tries[k_try]
            mujoco.mj_setState(model, data, state, sig)
            model.actuator_gainprm[:, 0] = kp0 * (1 + 0.5 * level)
            model.actuator_biasprm[:, 1] = -kp0 * (1 + 0.5 * level)
            closing.clear()
            ik_cache.clear()
            if level > 0:  # fingertips of each hand LIFTING an object closed onto fixed points of it
                for s_ in sides:
                    best_o, best_n = None, 0
                    for o in objs:
                        nh = len(set(rep.get(o["name"], {}).get("hand_contact", {}).get(s_, [])) & set(seg))
                        if nh > best_n:
                            best_o, best_n = o, nh
                    if best_o is None:
                        continue
                    mine = sorted(set(rep[best_o["name"]]["hand_contact"][s_]) & set(seg))
                    Pn = surf[best_o["name"]][1]
                    lifted = [t_ for t_ in mine if t_ in best_o["W"] and
                              ((Pn @ best_o["W"][t_][:3, :3].T + best_o["W"][t_][:3, 3]) @ up).min() - table_h > 0.01]
                    if not lifted:
                        continue  # resting on the table: closing would only push it
                    onset = lifted[0]
                    cands = [t_ for t_ in mine if onset - a.pregrasp <= t_ <= onset]
                    plans = [(len(pl), t_, pl) for t_ in cands for pl in [plan_tips(s_, t_, best_o)]]
                    n_, tp_, tips = max(plans, key=lambda x: (x[0], x[1])) if plans else (0, None, {})
                    if n_ >= 2:
                        # close only over the lift itself: from just before it leaves the table until it is down
                        # (closing during the approach or while it rests only pushes it)
                        # close from --pregrasp frames BEFORE the lift (a person closes the hand before lifting;
                        # closing only once the wrist rises is too late) until the object is put down
                        closing[s_] = {"obj": best_o, "tips": tips, "squeeze": a.close_step * level,
                                       "frames": set(range(onset - a.pregrasp, lifted[-1] + 3)), "plan_frame": tp_}
                        print(f"    {s_} lifts {best_o['name']} from frame {onset}: {n_} fingertip targets "
                              f"planned at frame {tp_}", flush=True)
            del frames_out[n_rec:]
            errs, per = [], {}
            for k in range(i, j_end + 1):
                step_frame(ts[k], ts[min(k + 1, len(ts) - 1)])
                record(ts[k])
                errs.append(obj_err(ts[k]))
                if a.debug_frames and level == 2 and ts[k] in a.debug_frames:
                    for s_, c_ in closing.items():
                        if ts[k] not in c_["frames"]:
                            continue
                        bid_ = obj_bid[c_["obj"]["name"]]
                        Wo = np.eye(4)
                        Wo[:3, :3], Wo[:3, 3] = data.xmat[bid_].reshape(3, 3), data.xpos[bid_]
                        rest_ = np.array(prep["sides"][s_]["rest_joints"])
                        dd = []
                        for tj, (p_, n_) in c_["tips"].items():
                            jb = [c for c in CHAINS if c[-1] == tj][0][2]
                            b_ = model.body(f"{s_}_j{jb}").id
                            tip_w = data.xpos[b_] + data.xmat[b_].reshape(3, 3) @ (rest_[tj] - rest_[jb])
                            tgt = Wo[:3, :3] @ (p_ + n_ * (RAD[tj] - c_["squeeze"])) + Wo[:3, 3]
                            dd.append(f"{tj}:{100 * np.linalg.norm(tip_w - tgt):.1f}")
                        terr = np.abs(data.qpos[hinge_q[s_]] - data.ctrl[act[s_]])
                        qa_ = root_q[s_]
                        w_, x_, y_, z_ = data.qpos[qa_ + 3:qa_ + 7]
                        Rr = Rotation.from_quat([x_, y_, z_, w_]).as_matrix()
                        qh = data.qpos[hinge_q[s_]]
                        Jm = fk_all(s_, Rr, data.qpos[qa_:qa_ + 3], {j: qh[3 * kk:3 * kk + 3] for kk, j in enumerate(ACT_J)})
                        Jc = fk_all(s_, Rr, data.qpos[qa_:qa_ + 3], {j: data.ctrl[act[s_]][3 * kk:3 * kk + 3] for kk, j in enumerate(ACT_J)})
                        b8 = model.body(f"{s_}_j7").id
                        tip8 = data.xpos[b8] + data.xmat[b8].reshape(3, 3) @ (rest_[8] - rest_[7])
                        b0 = model.body(f"{s_}_root").id
                        print(f"      f{ts[k]} {s_} index tip: model(sim q) vs MuJoCo {100 * np.linalg.norm(Jm[8] - tip8):.2f} cm, "
                              f"model(ctrl) vs MuJoCo {100 * np.linalg.norm(Jc[8] - tip8):.2f} cm, root body vs qpos "
                              f"{100 * np.linalg.norm(data.xpos[b0] - data.qpos[qa_:qa_ + 3]):.2f} cm", flush=True)
                        print(f"      f{ts[k]} {s_} tip-to-target cm {' '.join(dd)}  contacts {contact_log.get(ts[k])}  "
                              f"servo err mean {terr.mean():.3f} max {terr.max():.3f} rad  "
                              f"force {np.abs(data.actuator_force[act[s_]]).max():.2f}", flush=True)
                if False:  # debug trace of the ep11 lift
                    print(f"      L{level} f{ts[k]} contacts {contact_log.get(ts[k])} ctrl-ref "
                          f"{ {s_: round(float(np.abs(data.ctrl[act[s_]] - ref_hand(s_, ts[k])[2]).max()), 3) for s_ in sides if ref_hand(s_, ts[k]) is not None} }",
                          flush=True)
                for nm, e in obj_errs(ts[k]).items():
                    per.setdefault(nm, []).append(e)
            worst = max(errs)
            print(f"  grasp {seg[0]}-{seg[-1]} level {level}: " + ", ".join(
                f"{nm} mean {np.mean(v):.1f} max {np.max(v):.1f} cm" for nm, v in per.items()) +
                f"  closing {sorted(closing)}", flush=True)
            if best is None or np.mean(errs) < best[0]:
                best = (float(np.mean(errs)), level)
            k_try += 1
            if worst <= a.slip_cm or k_try > len(tries):
                break  # held, or the best attempt has just been replayed
            if k_try == len(tries):  # nothing held: replay the best attempt so it is what gets recorded
                if best[1] == level:
                    break
                tries.append(best[1])
        levels.append({"frames": [seg[0], seg[-1]], "grip_level": level, "squeeze_mm": 1000 * a.close_step * level,
                       "worst_err_cm": round(worst, 2)})
        print(f"grasp {seg[0]}-{seg[-1]}: grip level {level}, worst object error {worst:.1f} cm", flush=True)
        model.actuator_gainprm[:, 0] = kp0
        model.actuator_biasprm[:, 1] = -kp0
        closing.clear()
        i = j_end + 1

    # ------------------------------------------------------------------ outputs
    report = {"objects": {}, "grasps": levels, "max_hand_object_penetration_mm": pen, "args": vars(a),
              "contacts_per_frame": {str(t): v for t, v in contact_log.items()},
              "wrist_err_cm": {s_: {str(t): e for t, e in d.items()} for s_, d in wrist_err.items()}}
    for o in objs:
        nm = o["name"]
        e = np.array([np.linalg.norm(sim_obj[nm][t][:3, 3] - o["W"][t][:3, 3]) * 100 for t in ts if t in o["W"]])
        report["objects"][nm] = {"mean_cm": float(e.mean()), "max_cm": float(e.max()), "frac_lt2": float((e < 2).mean()),
                                 "frac_lt3": float((e < 3).mean()), "err_cm_per_frame": [round(float(v), 2) for v in e]}
        pdir = run / "sim_mj" / nm / "poses"
        pdir.mkdir(parents=True, exist_ok=True)
        for t, M in sim_obj[nm].items():
            C = np.linalg.inv(slam[t]) @ M
            (pdir / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
        print(f"{a.run} {nm:18s} sim vs reconstruction: mean {e.mean():.1f} cm, max {e.max():.1f} cm, "
              f"<2cm {100 * (e < 2).mean():.0f}%", flush=True)
    print(f"{a.run}: max hand-object penetration in simulation {pen:.1f} mm", flush=True)
    for s in sides:  # simulated hands back to the MANO schema (camera frame)
        sp = prep["sides"][s]
        M = np.eye(3) if sp["is_right"] else F
        rest = np.array(sp["rest_joints"])
        order = {"index_middle_pinky_ring_thumb": [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3],
                 "index_middle_ring_pinky_thumb": [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 1, 2, 3]}[sp["pose_order"]]
        od = run / "stereo" / a.out / s
        od.mkdir(parents=True, exist_ok=True)
        for t, (Rw, pw, q) in sim_hand[s].items():
            if str(t) not in sp["frames"]:
                continue
            S = slam[t]
            Rc = S[:3, :3].T @ Rw
            pc = S[:3, :3].T @ (pw - S[:3, 3])
            hin = {j: q[3 * k:3 * k + 3] for k, j in enumerate(ACT_J)}
            hp = [Rotation.from_matrix(M @ Rotation.from_euler("XYZ", hin[j]).as_matrix() @ M).as_rotvec() for j in order]
            go = Rotation.from_matrix(M @ Rc @ M).as_rotvec()
            # simulated joint positions (forward kinematics of the simulated hand), camera frame
            J = np.zeros((21, 3))
            Rcum = {0: Rc}
            J[0] = pc
            for j in range(1, 21):
                pa = PARENT[j]
                J[j] = J[pa] + Rcum[pa] @ (rest[j] - rest[pa])
                Rcum[j] = Rcum[pa] @ (Rotation.from_euler("XYZ", hin[j]).as_matrix() if j in hin else np.eye(3))
            rec = {"mano": {"betas": sp["betas"], "global_orient": go.tolist(), "hand_pose": np.ravel(hp).tolist()},
                   "is_right": sp["is_right"], "T": (pc - rest[0]).tolist(), "joints": J.tolist(), "source": "mujoco"}
            (od / f"{t:06d}.json").write_text(json.dumps(rec))
    (run / "stereo" / a.out / "report.json").write_text(json.dumps(report, indent=1))

    if frames_out:
        import cv2
        import imageio
        out = Path(os.environ.get("V2D_VIZ", ROOT / "viz")) / "10_mujoco"
        out.mkdir(parents=True, exist_ok=True)
        wr = imageio.get_writer(str(out / f"{a.run}.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=2)
        errs = {o["name"]: dict(zip([t for t in ts if t in o["W"]], report["objects"][o["name"]]["err_cm_per_frame"]))
                for o in objs}
        for t, img in frames_out:
            img = np.ascontiguousarray(img)
            cv2.putText(img, f"{a.run}  frame {t}  MuJoCo  (green ghost = reconstructed object)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            for k, (nm, d) in enumerate(errs.items()):
                e = d.get(t)
                if e is not None:
                    cv2.putText(img, f"{nm}: {e:.1f} cm", (8, 46 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (60, 230, 60) if e < 3 else (240, 60, 60), 2)
            wr.append_data(img)
        wr.close()
        print(f"wrote {out / (a.run + '.mp4')}", flush=True)


if __name__ == "__main__":
    main()
