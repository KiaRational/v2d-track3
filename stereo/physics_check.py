"""Physics check of the reconstructed hand-object interaction in MuJoCo (venv-mj, CPU).

Question: if the reconstructed hands move exactly as reconstructed, does physics move the
object the way we reconstructed it? A reference that fails here -- object dropped, flung by
a penetrating finger, left behind because no finger touches it -- is one the policy stage
(FlashCHORD, MuJoCo Warp contacts) cannot learn to reproduce.

Scene, in the SLAM world frame (gravity along the head camera's mean "down"):
  table    static box; its top at the lowest object vertex in the objects' first frames
  objects  free rigid bodies, collision = CoACD convex decomposition of the scanned mesh (keeps
           cups/pots hollow), density --density, friction --friction; start at their
           reconstructed first pose
  hands    kinematic (mocap) capsules along the 20 bones of each fitted MANO skeleton plus 5
           palm capsules, driven along the reconstruction and interpolated between frames;
           they push objects but ignore the table
Per object: simulated vs reconstructed position error per frame, drop/fling events (error
jumps), hand contacts per frame, deepest hand-object contact penetration.
Writes stereo/physics_<hands>.json and the simulated trajectory as a variant
<run>/sim_<hands>/<obj>/poses/NNNNNN.json (camera frame), renderable with render_4d_scene.py.
Usage: python physics_check.py <run> [...] [--hands mano_fit] [--variant fused_all_sam3]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import coacd
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
         (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
         (0, 5), (0, 9), (0, 13), (0, 17), (5, 17)]  # last five: palm
# MANO finger radii (thumb ~9 mm, fingers ~7.5 mm, palm slab ~11 mm): fatter capsules would add
# penetration that the reconstruction does not have
RADIUS = [0.009] * 4 + [0.0075] * 12 + [0.011] * 5
CACHE = ROOT / "cache/coacd"


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


def convex_parts(name):
    """CoACD decomposition of the scanned mesh, cached as part_*.obj."""
    d = CACHE / name
    parts = sorted(d.glob("part_*.obj")) if d.is_dir() else []
    if parts:
        return parts
    d.mkdir(parents=True, exist_ok=True)
    m = trimesh.load(ROOT / "meshes_obj" / name / f"{name}.obj", force="mesh")
    res = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=0.04, max_convex_hull=24)
    for i, (v, f) in enumerate(res):
        trimesh.Trimesh(v, f).export(d / f"part_{i:02d}.obj")
    return sorted(d.glob("part_*.obj"))


def quat_z_to(v):
    """MuJoCo quaternion (w,x,y,z) rotating +z onto v."""
    v = v / max(np.linalg.norm(v), 1e-9)
    r, _ = Rotation.align_vectors([v], [[0, 0, 1.0]])
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


def urdf_inertial(name):
    """Mass/COM/inertia from the dataset URDF -- what the policy simulator uses (every Track 3 object is a
    0.3 kg placeholder with a surface-shell inertia estimate). None if the URDF is missing."""
    import re
    u = ROOT / "data/track_3/public/urdf" / f"{name}_rigid.urdf"
    if not u.exists():  # objects with no provided scan (wooden_spoon): our own URDF, same 0.3 kg placeholder
        u = ROOT / "urdf_extra" / f"{name}_rigid.urdf"
    if not u.exists():
        return None
    x = u.read_text()
    m = float(re.search(r'<mass value="([^"]+)"', x).group(1))
    com = [float(v) for v in re.search(r'<inertial>\s*<origin xyz="([^"]+)"', x).group(1).split()]
    I = {k: float(re.search(rf'{k}="([^"]+)"', x).group(1)) for k in ("ixx", "iyy", "izz")}
    return m, com, [I["ixx"], I["iyy"], I["izz"]]


def build(objs, up, table_top, table_c, fps, sub, density, friction):
    gx, gy, gz = -9.81 * up
    xml = [f'<mujoco><option timestep="{1.0 / (fps * sub):.6f}" gravity="{gx} {gy} {gz}" cone="elliptic" '
           f'impratio="10"/>',
           f'<default><geom friction="{friction} 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>',
           "<asset>"]
    for o in objs:
        for k, p in enumerate(o["parts"]):
            xml.append(f'<mesh name="{o["name"]}_{k}" file="{p}"/>')
    xml.append("</asset><worldbody>")
    tq = quat_z_to(up)
    tc = table_c - up * 0.02
    xml.append(f'<geom name="table" type="box" size="1.0 1.0 0.02" pos="{tc[0]} {tc[1]} {tc[2]}" '
               f'quat="{tq[0]} {tq[1]} {tq[2]} {tq[3]}" contype="1" conaffinity="1"/>')
    for o in objs:
        W = o["W0"]
        q = Rotation.from_matrix(W[:3, :3]).as_quat()
        xml.append(f'<body name="{o["name"]}" pos="{W[0, 3]} {W[1, 3]} {W[2, 3]}" quat="{q[3]} {q[0]} {q[1]} {q[2]}">'
                   f'<freejoint/>')
        ine = urdf_inertial(o["name"])
        if ine is not None:  # the training simulator's mass properties, not a filled-hull density
            (mass, com, dI) = ine
            xml.append(f'<inertial pos="{com[0]} {com[1]} {com[2]}" mass="{mass}" diaginertia="{dI[0]} {dI[1]} {dI[2]}"/>')
        dens = 0 if ine is not None else density
        for k in range(len(o["parts"])):
            xml.append(f'<geom type="mesh" mesh="{o["name"]}_{k}" density="{dens}" contype="1" conaffinity="3" '
                       f'condim="4"/>')
        xml.append("</body>")
    for side in ("left", "right"):
        for b, (i, j) in enumerate(BONES):
            xml.append(f'<body name="{side}_b{b}" mocap="true" pos="0 0 -10"><geom type="capsule" '
                       f'size="{RADIUS[b]} 0.01" contype="2" conaffinity="0" rgba="0.9 0.7 0.6 1"/></body>')
    xml.append("</worldbody></mujoco>")
    return mujoco.MjModel.from_xml_string("\n".join(xml))


def stability(a, r, run, model, data, objs, bid, H, ts, up, table_top, set_hands, meta):
    """Per grasp frame: hands frozen, object released at its pose -- does the grasp hold it?"""
    rep = json.loads((run / "fusion_report.json").read_text())
    out = {}
    for o in objs:
        nm = o["name"]
        held = sorted(set().union(*[set(v) for v in rep.get(nm, {}).get("hand_contact", {}).values()] or [set()]))
        lift = []
        for t in held[::a.stride]:
            if t not in o["W"]:
                continue
            # only frames where the object is off the table: on the table, the table holds it
            if ((o["V"] @ o["W"][t][:3, :3].T + o["W"][t][:3, 3]) @ up).min() - table_top < 0.01:
                continue
            mujoco.mj_resetData(model, data)
            for p in objs:  # the other objects out of the way, this one at its pose
                j = model.body(p["name"]).jntadr[0]
                qa = model.jnt_qposadr[j]
                W = o["W"][t] if p is o else np.eye(4)
                pos = W[:3, 3] if p is o else np.array([0, 0, -50.0 - objs.index(p)])
                q = Rotation.from_matrix(W[:3, :3]).as_quat()
                data.qpos[qa:qa + 7] = [*pos, q[3], q[0], q[1], q[2]]
            set_hands(t, 0, t)
            mujoco.mj_forward(model, data)
            p0 = data.xpos[bid[nm]].copy()
            for _ in range(int(a.hold * meta["fps"] * a.sub)):
                mujoco.mj_step(model, data)
            lift.append((t, float(np.linalg.norm(data.xpos[bid[nm]] - p0) * 100)))
        d = np.array([x for _, x in lift])
        row = {"lifted_grasp_frames": len(lift), "mean_disp_cm": float(d.mean()) if len(d) else None,
               "stable_frac_2cm": float((d < 2).mean()) if len(d) else None,
               "per_frame": [{"t": t, "disp_cm": round(x, 2)} for t, x in lift]}
        out[nm] = row
        f = lambda x: "   -" if x is None else f"{x:5.2f}"  # noqa: E731
        print(f"{r:18s} {nm:18s} lifted grasp frames {len(lift):4d}  mean fall {f(row['mean_disp_cm'])} cm  "
              f"stable(<2cm) {f(row['stable_frac_2cm'])}", flush=True)
    (run / "stereo" / f"stability_{a.hands}.json").write_text(json.dumps({"hands": a.hands, "objects": out}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", default="mano_fit")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--sub", type=int, default=10, help="physics substeps per video frame")
    ap.add_argument("--density", type=float, default=400.0)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--no_hands", action="store_true", help="sanity check: objects alone must stay put")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds of settling on the table before playback")
    ap.add_argument("--stability", action="store_true",
                    help="per-frame grasp stability instead of replay: freeze the hands at a grasp frame, put the "
                         "object at its reconstructed pose, simulate --hold s of gravity, measure how far it falls")
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    print(f"{'run':18s} {'object':18s} {'mean_cm':>7s} {'max_cm':>7s} {'<3cm':>5s} {'<5cm':>5s} {'drops':>5s} "
          f"{'contact%':>8s} {'pen_mm':>6s}", flush=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        meta = json.loads((run / "episode.json").read_text())
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        ts = sorted(slam)
        up = -np.mean([slam[t][:3, 1] for t in ts], 0)  # fallback only: tilted by the head pitch
        up /= np.linalg.norm(up)
        # hands: world-frame joints per frame
        H = {}
        for side in ("left", "right"):
            for f in (run / "stereo" / a.hands / side).glob("*.json"):
                t = int(f.stem)
                if t in slam:
                    J = np.array(json.loads(f.read_text())["joints"])
                    H.setdefault(side, {})[t] = J @ slam[t][:3, :3].T + slam[t][:3, 3]
        objs = []
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            pd = run / a.variant / o["name"] / "poses"
            W = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in ts if (pd / f"{t:06d}.json").exists()}
            if not W:
                continue
            m = trimesh.load(ROOT / "meshes_obj" / o["name"] / f"{o['name']}.obj", force="mesh")
            objs.append({"name": o["name"], "W": W, "W0": W[min(W)], "parts": convex_parts(o["name"]),
                         "V": np.asarray(m.vertices)})
        low = [((o["V"] @ o["W0"][:3, :3].T + o["W0"][:3, 3]) @ up).min() for o in objs]
        table_top = min(low)
        tp = run / "table_plane.json"
        if tp.exists():  # measured support plane (stereo RANSAC): true gravity + table height
            d = json.loads(tp.read_text())
            up, table_top = np.array(d["up"]), float(d["height"])
        cen = np.mean([o["W0"][:3, 3] for o in objs], 0)
        table_c = cen - up * (cen @ up - table_top)
        model = build(objs, up, table_top, table_c, meta["fps"], a.sub, a.density, a.friction)
        data = mujoco.MjData(model)
        bid = {o["name"]: model.body(o["name"]).id for o in objs}
        mids = {s: [model.body(f"{s}_b{b}").mocapid[0] for b in range(len(BONES))] for s in ("left", "right")}
        hand_geoms = {model.body(f"{s}_b{b}").id for s in ("left", "right") for b in range(len(BONES))}

        def bone_poses(J):
            out = []
            for b, (i, j) in enumerate(BONES):
                pa, pb = J[i], J[j]
                out.append(((pa + pb) / 2, quat_z_to(pb - pa), max(np.linalg.norm(pb - pa) / 2 - RADIUS[b], 1e-3)))
            return out

        # capsule half-lengths from the median bone length of the sequence (MANO bones are rigid)
        for s, seq in H.items():
            L = np.median([[p[2] for p in bone_poses(J)] for J in seq.values()], 0)
            for b in range(len(BONES)):
                model.geom_size[model.body(f"{s}_b{b}").geomadr[0], 1] = L[b]

        def set_hands(t, al, t2):
            for s in ("left", "right"):
                seq = {} if a.no_hands else H.get(s, {})
                if t in seq:
                    J = seq[t] if (t2 not in seq or al == 0) else (1 - al) * seq[t] + al * seq[t2]
                    for b, (p, q, _) in enumerate(bone_poses(J)):
                        data.mocap_pos[mids[s][b]] = p
                        data.mocap_quat[mids[s][b]] = q
                else:
                    for b in range(len(BONES)):
                        data.mocap_pos[mids[s][b]] = [0, 0, -10]

        if a.stability:
            stability(a, r, run, model, data, objs, bid, H, ts, up, table_top, set_hands, meta)
            continue
        t0 = ts[0]
        # settle the objects on the table with no hands, then bring the hands in
        for b in range(len(BONES)):
            for s_ in ("left", "right"):
                data.mocap_pos[mids[s_][b]] = [0, 0, -10]
        mujoco.mj_forward(model, data)
        for _ in range(int(a.settle * meta["fps"] * a.sub)):
            mujoco.mj_step(model, data)
        settle_err = {o["name"]: float(np.linalg.norm(data.xpos[bid[o["name"]]] - o["W0"][:3, 3]) * 100) for o in objs}
        print(f"{r}: settle drift (cm) {  {k: round(v, 2) for k, v in settle_err.items()} }", flush=True)
        set_hands(t0, 0, t0)
        sim = {o["name"]: {} for o in objs}
        cont = {o["name"]: 0 for o in objs}
        pen = {o["name"]: 0.0 for o in objs}
        for k, t in enumerate(ts):
            t2 = ts[k + 1] if k + 1 < len(ts) else t
            for s_ in range(a.sub):
                set_hands(t, s_ / a.sub, t2)
                mujoco.mj_step(model, data)
            touched = set()
            for c in data.contact[:data.ncon]:
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                for o in objs:
                    if bid[o["name"]] in (b1, b2) and (b1 in hand_geoms or b2 in hand_geoms):
                        touched.add(o["name"])
                        pen[o["name"]] = max(pen[o["name"]], -c.dist * 1000)
            for o in objs:
                p = data.xpos[bid[o["name"]]].copy()
                R = data.xmat[bid[o["name"]]].reshape(3, 3).copy()
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = R, p
                sim[o["name"]][t] = M
                cont[o["name"]] += o["name"] in touched
        out = {}
        for o in objs:
            nm = o["name"]
            e = np.array([np.linalg.norm(sim[nm][t][:3, 3] - o["W"][t][:3, 3]) * 100 for t in ts if t in o["W"]])
            drops = int((np.diff(e) > 2.0).sum()) if len(e) > 1 else 0
            row = {"mean_cm": float(e.mean()), "max_cm": float(e.max()), "frac_lt3": float((e < 3).mean()),
                   "frac_lt5": float((e < 5).mean()), "jumps_gt2cm_per_frame": drops,
                   "contact_frac": cont[nm] / len(ts), "max_contact_pen_mm": pen[nm],
                   "err_cm_per_frame": [round(float(x), 2) for x in e]}
            out[nm] = row
            pdir = run / f"sim_{'nohands' if a.no_hands else a.hands}" / nm / "poses"
            pdir.mkdir(parents=True, exist_ok=True)
            for t, M in sim[nm].items():
                C = np.linalg.inv(slam[t]) @ M
                (pdir / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            print(f"{r:18s} {nm:18s} {row['mean_cm']:7.1f} {row['max_cm']:7.1f} {row['frac_lt3']:5.2f} "
                  f"{row['frac_lt5']:5.2f} {drops:5d} {100 * row['contact_frac']:7.0f}% {pen[nm]:6.1f}", flush=True)
        tag = "nohands" if a.no_hands else a.hands
        (run / "stereo" / f"physics_{tag}.json").write_text(json.dumps(
            {"hands": a.hands, "variant": a.variant, "params": vars(a), "objects": out}, indent=1))


if __name__ == "__main__":
    main()
