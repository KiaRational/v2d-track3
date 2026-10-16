"""Synthetic tests of the joint hand-object optimisation (stereo/s06j_joint_hoi.py). No run data, no MANO:
a rigid point-cloud "hand" (778 vertices, 21 joints, joint 0 = wrist at the origin) and an analytic box SDF."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "stereo")]
os.environ.setdefault("V2D_ROOT", tempfile.gettempdir())
import s06j_joint_hoi as J  # noqa: E402

N, FPS = 60, 20.0
HALF = np.array([0.03, 0.02, 0.04])            # box half extents (no symmetry but the 180 deg flips)
G_T = np.array([0.0, 0.0, 0.08])                # object centre in the wrist frame (box face at z = 4 cm: contact)
MIR = np.diag([-1.0, 1.0, 1.0])
_rng = np.random.default_rng(0)
TV = _rng.uniform([-.02, -.02, 0], [.02, .02, .04], (778, 3))  # the top layer touches the held box face
TJ = _rng.uniform([-.02, -.02, 0], [.02, .02, .03], (21, 3))
TJ[0] = 0


def fk(go, hp, be, is_right):
    R = J.rotvec_to_matrix(go)
    tv, tj = torch.tensor(TV, dtype=R.dtype), torch.tensor(TJ, dtype=R.dtype)
    v = torch.einsum("bij,nj->bni", R, tv) + 1e-3 * hp.mean(-1)[:, None, None]
    j = torch.einsum("bij,nj->bni", R, tj) + 1e-3 * hp.mean(-1)[:, None, None]
    if not is_right:
        m = torch.tensor([-1.0, 1.0, 1.0])
        v, j = v * m, j * m
    return v, j


class Box:
    def __init__(self, *args, **kwargs):
        self.h = torch.tensor(HALF, dtype=torch.float32)

    def __call__(self, p):
        q = p.abs() - self.h
        return q.clamp(min=0).norm(dim=-1) + q.max(-1).values.clamp(max=0)


def pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def hand_rec(Rh, ph, is_right):
    """World wrist pose (identity camera) -> MANO-like record arrays (go, T, joints)."""
    Rgo = Rh if is_right else MIR @ Rh @ MIR
    go = Rotation.from_matrix(Rgo).as_rotvec()
    j = (TJ @ Rgo.T) * (1 if is_right else MIR.diagonal()) + ph
    return go, ph, j


def hand_entry(Rs, ps, is_right, conf=0.9, bias=None):
    """ref = observation (= truth + bias), tri = the same joints, confidence per frame (scalar or (n,))."""
    go, T, joints = [], [], []
    for t in range(N):
        p = ps[t] + (bias[t] if bias is not None else 0)
        g, tt, j = hand_rec(Rs[t], p, is_right)
        go.append(g), T.append(tt), joints.append(j)
    conf = np.broadcast_to(np.asarray(conf, float), (N,))
    return {"ts": list(range(N)), "is_right": is_right,
            "ref": {"go": np.array(go), "hp": np.zeros((N, 45)), "be": np.zeros((N, 10)), "T": np.array(T)},
            "init": None, "Sw": np.tile(np.eye(4), (N, 1, 1)), "tri": np.array(joints),
            "conf": np.repeat(conf[:, None], 21, 1)}


def obj_entry(W0, meas, grasp, static_w=None, sym=None):
    return {"name": "box", "W0": W0, "meas": meas, "sym": sym or {"group": [np.eye(3)], "cont_axis": None, "center": np.zeros(3)},
            "sdf": Box(), "verts": np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * HALF,
            "static_w": np.zeros(N - 1) if static_w is None else static_w, "grasp": grasp,
            "radius": float(np.linalg.norm(HALF))}


def wrist_path(t0=15, t1=44, start=(0.0, 0.0, 0.1), bend=0.06, approach=(0, 0, -0.004)):
    """Hand approaches (along its -z), carries the object on a curve during [t0, t1], retreats."""
    Rs, ps = [], []
    for t in range(N):
        s = np.clip((t - t0) / (t1 - t0), 0, 1)
        s = s * s * (3 - 2 * s)                    # smooth start/stop: real objects do not jerk
        R = Rotation.from_rotvec([0.3 * s, 0.2 * np.sin(2 * s), 0.1]).as_matrix()
        p = np.array(start) + [0.12 * s, bend * np.sin(np.pi * s), 0.03 * s]
        if t < t0:
            p = p + R @ np.array(approach) * 0.1 * (t0 - t) ** 2  # decelerating approach
        if t > t1:
            p = p + R @ np.array(approach) * 0.1 * (t - t1) ** 2
        Rs.append(R), ps.append(p)
    return np.array(Rs), np.array(ps)


def object_truth(Rs, ps, t0=15, t1=44):
    W = []
    for t in range(N):
        k = min(max(t, t0), t1)
        W.append(pose(Rs[k], ps[k]) @ pose(np.eye(3), G_T))
    return np.array(W)


def noisy(T, st, sr, rng):
    M = T.copy()
    M[:3, 3] += rng.normal(0, st, 3)
    M[:3, :3] = Rotation.from_rotvec(rng.normal(0, sr, 3)).as_matrix() @ M[:3, :3]
    return M


def cfg(**kw):
    base = dict(device="cpu", iters_stage1=150, iters_stage2=350, lr=1e-2)
    base.update(kw)
    return J.default_cfg(**base)


def centroid_err_mm(P, W, frames):
    return np.array([np.linalg.norm(P[t][:3, 3] - W[t][:3, 3]) for t in frames]) * 1000


def solved_poses(res):
    f = res["objf"][0]
    return np.array([pose(R, p) for R, p in zip(f["R"].detach().numpy(), f["p"].detach().numpy())])


def wrist_world(res, side):
    return res["hf"][side]["Hp"].detach().numpy()


class JointTests(unittest.TestCase):
    def test_coupling_recovers_occluded_object_from_moving_hand(self):
        rng = np.random.default_rng(1)
        Rs, ps = wrist_path()
        W = object_truth(Rs, ps)
        gap = range(22, 38)
        meas = [(t, noisy(W[t], .002, np.deg2rad(1), rng), 1.0, 1.0) for t in range(N) if t not in gap]
        W0 = W.copy()
        for t in gap:
            W0[t] = W[21]                       # a fuser without the hand freezes the hidden object
        hands = {"right": hand_entry(Rs, ps, True)}
        # the hidden carry moves ~6 cm away from the frozen init: beyond the default 40 mm object trust limit
        P, res = J.solve(hands, [obj_entry(W0, meas, {"right": list(range(15, 45))})], None, FPS, N, cfg(max_obj_shift_mm=250), fk)
        e0, e1 = centroid_err_mm(W0, W, gap), centroid_err_mm(solved_poses(res), W, gap)
        print(f"occluded carry: init {e0.mean():.1f} mm -> joint {e1.mean():.1f} mm (max {e1.max():.1f})")
        self.assertGreater(e0.mean(), 15)
        self.assertLess(e1.mean(), 3.0)
        self.assertLess(e1.mean(), 0.15 * e0.mean())
        self.assertTrue(P.segs and P.segs[0]["side"] == "right")

    def test_object_track_corrects_biased_wrist(self):
        rng = np.random.default_rng(2)
        Rs, ps = wrist_path()
        W = object_truth(Rs, ps)
        bias = np.zeros((N, 3))
        bias[25:35] = [0.015, 0.0, 0.0]         # the stereo hand is off for 10 frames of the grasp
        conf = np.full(N, 0.9)
        conf[25:35] = 0.1
        meas = [(t, noisy(W[t], .001, np.deg2rad(.5), rng), 5.0, 5.0) for t in range(N)]
        hands = {"right": hand_entry(Rs, ps, True, conf=conf, bias=bias)}
        P, res = J.solve(hands, [obj_entry(W.copy(), meas, {"right": list(range(15, 45))})], None, FPS, N, cfg(), fk)
        e0 = np.linalg.norm(bias[25:35], axis=1).mean() * 1000
        e1 = np.linalg.norm(wrist_world(res, "right")[25:35] - ps[25:35], axis=1).mean() * 1000
        eo = centroid_err_mm(solved_poses(res), W, range(25, 35)).mean()
        print(f"biased wrist: {e0:.1f} mm -> {e1:.1f} mm (object moved {eo:.1f} mm)")
        # partial by design: with the object 8 cm from the wrist the coupling can also be met by rotating the hand,
        # which s06h's prior prices ~400x below translating it (w_trans 2e4 vs w_rot 50)
        self.assertLess(e1, 0.75 * e0)
        self.assertLess(eo, 4.0)
        # a constant bias over the whole grasp is absorbed by the grasp offset G (unobservable by coupling)
        clean = np.linalg.norm(wrist_world(res, "right")[:20] - ps[:20], axis=1).mean() * 1000
        self.assertLess(clean, 3.0)

    def test_handover_is_continuous(self):
        rng = np.random.default_rng(3)
        t_sw = 30
        RsL, psL = wrist_path(t0=10, t1=t_sw, start=(0.0, 0.0, 0.1))
        # object path: carried by the left hand until t_sw, then by the right hand (its own grasp offset)
        W = object_truth(RsL, psL, 10, t_sw)
        Gr = pose(Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix(), [0.0, 0.0, 0.07])  # face at 4 cm
        Rr, pr = [], []
        for t in range(N):
            s = np.clip((t - t_sw) / 20, 0, 1)
            s = s * s * (3 - 2 * s)
            Wo = W[t_sw].copy()
            Wo[:3, 3] += [0.0, 0.10 * s, 0.02 * np.sin(np.pi * s)]
            if t > t_sw:
                W[t] = Wo
            Hr = (Wo if t >= t_sw else W[t_sw]) @ np.linalg.inv(Gr)
            Hr[:3, 3] += Hr[:3, :3] @ [0, 0, -0.0004 * max(0, t_sw - t) ** 2]  # right hand approaches before t_sw
            Rr.append(Hr[:3, :3]), pr.append(Hr[:3, 3])
        for t in range(t_sw + 1, N):
            if t > 50:
                W[t] = W[50]
        Rr, pr = np.array(Rr), np.array(pr)
        for t in range(51, N):                   # right hand retreats after 50
            pr[t] = pr[50] + Rr[50] @ [0, 0, -0.0004 * (t - 50) ** 2]
        gap = range(24, 38)
        meas = [(t, noisy(W[t], .002, np.deg2rad(1), rng), 1.0, 1.0) for t in range(N) if t not in gap]
        W0 = W.copy()
        for t in gap:
            W0[t] = W[23]
        hands = {"left": hand_entry(RsL, psL, False), "right": hand_entry(Rr, pr, True)}
        grasp = {"left": list(range(10, t_sw + 1)), "right": list(range(t_sw + 1, 51))}
        P, res = J.solve(hands, [obj_entry(W0, meas, grasp)], None, FPS, N, cfg(max_obj_shift_mm=250), fk)
        S = solved_poses(res)
        step = np.linalg.norm(np.diff(S[:, :3, 3], axis=0), axis=1) * 1000
        step_gt = np.linalg.norm(np.diff(W[:, :3, 3], axis=0), axis=1) * 1000
        e1 = centroid_err_mm(S, W, gap)
        sw = step[t_sw - 3:t_sw + 3]
        print(f"handover: max step at switch {sw.max():.2f} mm (truth {step_gt[t_sw - 3:t_sw + 3].max():.2f}), "
              f"in-segment max {step[12:t_sw - 4].max():.2f} mm, gap error {e1.mean():.1f} mm")
        self.assertTrue(any(g.get("handover") for g in P.segs))
        self.assertLessEqual(sw.max(), max(step[12:t_sw - 4].max(), step[t_sw + 4:48].max()) + 1.0)
        self.assertLess(e1.mean(), 4.0)

    def test_no_hand_object_stays_still(self):
        rng = np.random.default_rng(4)
        Rs, ps = wrist_path(start=(0.4, 0.0, 0.1))          # the hand works far away (> 20 cm)
        W = np.tile(pose(np.eye(3), [0.0, 0.0, 0.1]), (N, 1, 1))
        meas = []
        for t in range(N):
            M = noisy(W[t], .001, np.deg2rad(.5), rng)
            w = 1.0
            if 20 <= t < 40:                                  # occluded: drifting, jittering tracker
                M[:3, 3] += [0.015 * np.sin(np.pi * (t - 20) / 20), 0, 0] + rng.normal(0, .004, 3)
                w = 0.4
            meas.append((t, M, w, w * w))
        hands = {"right": hand_entry(Rs, ps, True)}
        errs = {}
        for tag, w in (("off", 0.0), ("on", None)):
            c = cfg() if w is None else cfg(w_nohand=w)
            P, res = J.solve(hands, [obj_entry(W.copy(), meas, {})], None, FPS, N, c, fk)
            errs[tag] = centroid_err_mm(solved_poses(res), W, range(20, 40)).mean()
            if w is None:
                self.assertGreater(float(P.O[0]["nohand"].mean()), 0.9)
        print(f"no-hand stillness: without term {errs['off']:.2f} mm, with {errs['on']:.2f} mm")
        self.assertLess(errs["on"], 0.6 * errs["off"])

    def test_symmetry_and_se3(self):
        rng = np.random.default_rng(5)
        flip = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
        W = np.array([pose(Rotation.from_rotvec([0, 0, np.deg2rad(200) * t / (N - 1)]).as_matrix() @
                           Rotation.from_rotvec([0.4, 0, 0]).as_matrix(), [0.05 * t / N, 0, 0.1]) for t in range(N)])
        meas = []
        for t in range(N):
            M = noisy(W[t], .002, np.deg2rad(2), rng)
            if rng.random() < 0.5:                               # the estimator flips a symmetric box at random
                M[:3, :3] = M[:3, :3] @ flip
            meas.append((t, M, 1.0, 1.0))
        W0 = np.array([noisy(W[t], .003, np.deg2rad(3), rng) for t in range(N)])
        sym = {"group": [np.eye(3), flip], "cont_axis": None, "center": np.zeros(3)}
        P, res = J.solve({}, [obj_entry(W0, meas, {}, sym=sym)], None, FPS, N, cfg(w_nohand=0), fk)  # moves with no hand
        S = solved_poses(res)
        err = np.rad2deg(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", S[:, :3, :3], W[:, :3, :3])).as_rotvec(), axis=1))
        err0 = np.rad2deg(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", W0[:, :3, :3], W[:, :3, :3])).as_rotvec(), axis=1))
        print(f"discrete symmetry + 200 deg turn: rot err {err0.mean():.2f} -> {err.mean():.2f} deg")
        self.assertLess(err.mean(), 0.7 * err0.mean())
        print(f"  worst frame {int(err.argmax())}: {err.max():.2f} deg (init {err0[err.argmax()]:.2f}), p95 {np.percentile(err, 95):.2f}")
        self.assertLess(np.percentile(err, 95), 4.0)
        np.testing.assert_allclose(np.linalg.det(S[:, :3, :3]), 1.0, atol=1e-4)
        # continuous symmetry: the spin about the axis is unobservable -> held at the init track
        ax = np.array([0.0, 0.0, 1.0])
        Wc = np.array([pose(Rotation.from_rotvec([0.3 * np.sin(t / 10), 0, 0]).as_matrix(), [0, 0, 0.1]) for t in range(N)])
        mc = [(t, pose(Wc[t][:3, :3] @ Rotation.from_rotvec(ax * rng.uniform(-np.pi, np.pi)).as_matrix(), Wc[t][:3, 3]), 1.0, 1.0)
              for t in range(N)]
        W0c = np.array([pose(Wc[t][:3, :3] @ Rotation.from_rotvec(np.r_[rng.normal(0, np.deg2rad(3), 2), 0]).as_matrix(),
                             Wc[t][:3, 3]) for t in range(N)])  # init: axis tilt noise, no spin
        symc = {"group": [np.eye(3)], "cont_axis": ax, "center": np.zeros(3)}
        P, res = J.solve({}, [obj_entry(W0c, mc, {}, sym=symc)], None, FPS, N, cfg(w_nohand=0), fk)
        Sc = solved_poses(res)
        axe = np.rad2deg(np.arccos(np.clip(np.einsum("ni,ni->n", Sc[:, :3, :3] @ ax, Wc[:, :3, :3] @ ax), -1, 1)))
        axe0 = np.rad2deg(np.arccos(np.clip(np.einsum("ni,ni->n", W0c[:, :3, :3] @ ax, Wc[:, :3, :3] @ ax), -1, 1)))
        spin = np.rad2deg(np.abs(Rotation.from_matrix(np.einsum("nji,njk->nik", W0c[:, :3, :3], Sc[:, :3, :3])).as_rotvec() @ ax))
        print(f"continuous symmetry: axis err {axe0.mean():.2f} -> {axe.mean():.2f} deg, spin change {spin.max():.2f} deg")
        self.assertLess(axe.mean(), 0.7 * axe0.mean())
        self.assertLess(spin.max(), 3.0)

    def test_round_table_slab(self):
        from motion_quality import table_disk_inside, table_slab_depth
        disk = {"up": [0, 0, 1], "height": 0.0, "center": [0, 0, 0], "radius": 0.24, "thickness": 0.03}
        p = torch.tensor([[0.1, 0, -0.005],    # in the top, inside the disk: 5 mm
                          [0.1, 0, -0.025],    # near the underside: 5 mm to the bottom face
                          [0.1, 0, -0.10],     # under the table: free
                          [0.40, 0, -0.005],   # beside the table (outside the disk): free
                          [0.1, 0, 0.02]])     # above: free
        d = table_slab_depth(p, disk).numpy() * 1000
        np.testing.assert_allclose(d, [5, 5, 0, 0, 0], atol=1e-3)
        self.assertEqual(table_disk_inside(p, disk).tolist(), [True, True, True, False, True])
        legacy = table_slab_depth(p, {"up": [0, 0, 1], "height": 0.0}).numpy() * 1000
        np.testing.assert_allclose(legacy, [5, 25, 100, 5, 0], atol=1e-3)
        # side="auto": a hand above the table with a fingertip pressed 25 mm into the top is pushed UP
        # (depth 25 mm to the top face, not 5 mm to the bottom); a hand under it only out through the bottom
        above = torch.tensor([[[0.1, 0, 0.05], [0.1, 0, 0.04], [0.1, 0, -0.025]]])
        np.testing.assert_allclose(table_slab_depth(above, disk, side="auto").numpy()[0] * 1000, [0, 0, 25], atol=1e-3)
        below = torch.tensor([[[0.1, 0, -0.10], [0.1, 0, -0.09], [0.1, 0, -0.005]]])
        np.testing.assert_allclose(table_slab_depth(below, disk, side="auto").numpy()[0] * 1000, [0, 0, 25], atol=1e-3)
        g = above.clone().requires_grad_(True)
        (table_slab_depth(g, disk, side="auto") ** 2).sum().backward()
        self.assertLess(g.grad[0, 2, 2].item(), 0)  # descent (-grad) moves the pressed fingertip up
        # a hand passing under the disk is not pulled up through it by s06j
        Rs, ps = wrist_path(start=(0.0, 0.0, -0.12))
        hands = {"right": hand_entry(Rs, ps, True)}
        W = np.tile(pose(np.eye(3), [0.5, 0.0, 0.05]), (N, 1, 1))
        plane = {"up": [0, 0, 1], "height": 0.0, "center": [0, 0, 0], "radius": 0.24, "thickness": 0.03}
        P, res = J.solve(hands, [obj_entry(W, [(t, W[t], 1.0, 1.0) for t in range(N)], {})], plane, FPS, N,
                         cfg(iters_stage1=20, iters_stage2=60), fk)
        self.assertLess(np.abs(wrist_world(res, "right") - ps).max(), 1e-3)
        self.assertEqual(res["terms_after"]["table"], 0.0)

    def test_main_io_roundtrip(self):
        """End to end through main(): camera frames, output layout, report variant, overwrite guard."""
        import trimesh
        rng = np.random.default_rng(6)
        Rs, ps = wrist_path()
        W = object_truth(Rs, ps)
        cams = [pose(Rotation.from_rotvec([0.02 * t / N, 0.3, 0]).as_matrix(), [0.01 * t / N, 0.1, -0.2]) for t in range(N)]
        with tempfile.TemporaryDirectory() as tmp:
            root, run = Path(tmp), Path(tmp) / "runs/syn"

            def write(p, v):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(v))
            write(run / "episode.json", {"fps": FPS})
            write(run / "table_plane.json", {"up": [0, 0, 1], "height": -0.3, "center": [0, 0, -0.3], "radius": 0.24,
                                             "thickness": 0.03})
            write(run / "fusion_report.json", {"box": {"hand_contact": {"right": list(range(15, 45)), "left": []}}})
            write(run / "objects_sam3/objects.json", [{"id": 1, "name": "box"}])
            mesh = root / "meshes_obj/box/box.obj"
            mesh.parent.mkdir(parents=True)
            trimesh.creation.box(extents=2 * HALF).export(mesh)
            tri = {"right": {"X_smooth": {}, "conf": {}}}
            for t in range(N):
                Ci = np.linalg.inv(cams[t])
                write(run / f"recon_hawor/slam_poses/{t:06d}.json", {"R": cams[t][:3, :3].tolist(), "T": cams[t][:3, 3].tolist()})
                Oc = Ci @ W[t]
                write(run / f"fused_all_sam3/box/poses/{t:06d}.json", {"R": Oc[:3, :3].tolist(), "T": Oc[:3, 3].tolist()})
                Mc = Ci @ noisy(W[t], .001, np.deg2rad(.5), rng)
                write(run / f"objects_sam3/box/poses_smoothed/{t:06d}.json", {"R": Mc[:3, :3].tolist(), "T": Mc[:3, 3].tolist()})
                Hc = Ci @ pose(Rs[t], ps[t])                      # wrist in the camera frame
                go, T, j = hand_rec(Hc[:3, :3], Hc[:3, 3], True)
                rec = {"is_right": True, "T": T.tolist(), "joints": j.tolist(),
                       "mano": {"betas": [0] * 10, "global_orient": go.tolist(), "hand_pose": [0] * 45}}
                write(run / f"stereo/mano_fit/right/{t:06d}.json", rec)
                write(run / f"stereo/mano_fit_phys/right/{t:06d}.json", rec)
                tri["right"]["X_smooth"][str(t)], tri["right"]["conf"][str(t)] = j.tolist(), [0.9] * 21
            (run / "objects_sam3/box/stereo_icp").mkdir(parents=True)
            write(run / "stereo/hands_tri.json", tri)
            time.sleep(0.01)
            np.savez(run / "objects_sam3/box/.contact_vis_poses_smoothed.npz", near=np.zeros(N, bool),
                     vis=np.ones(N), touch=np.zeros(N, bool))
            argv = ["syn", "--device", "cpu", "--iters_stage1", "20", "--iters_stage2", "30"]
            with patch.object(J, "ROOT", root), patch.object(J, "Mano", lambda *a: None), \
                    patch.object(J, "SDFGrid", Box), patch.object(J, "mano_forward", lambda m, *a: fk(*a)), \
                    patch.dict(os.environ, {"V2D_ROOT": str(root)}):
                import fuse_objects
                with patch.object(fuse_objects, "ROOT", root):
                    J.main(argv)
                    rep = json.loads((run / "stereo/mano_fit_joint/report.json").read_text())
                    self.assertEqual(rep["args"]["variant"], "fused_joint")
                    self.assertIn("quality_passed", rep)
                    self.assertIn("right", rep["quality_summary"])
                    jr = json.loads((run / "fused_joint/joint_report.json").read_text())
                    self.assertIn("couple", jr["terms_after"])
                    self.assertEqual(len(jr["grasp_segments"]), 1)
                    for t in range(N):
                        d = json.loads((run / f"fused_joint/box/poses/{t:06d}.json").read_text())
                        Wj = cams[t] @ pose(np.array(d["R"]), d["T"])
                        self.assertLess(np.linalg.norm(Wj[:3, 3] - W[t][:3, 3]), 0.01)  # anchors pull the 1 cm gap
                        h = json.loads((run / f"stereo/mano_fit_joint/right/{t:06d}.json").read_text())
                        self.assertEqual(set(h), {"is_right", "T", "joints", "mano"})
                        m = h["mano"]
                        _, jj = fk(torch.tensor([m["global_orient"]]), torch.tensor([m["hand_pose"]]), None, True)
                        np.testing.assert_allclose(jj[0].numpy() + h["T"], h["joints"], atol=1e-5)
                    # outputs must never be an input, and a foreign dir is not overwritten
                    with self.assertRaises(ValueError):
                        J.main(argv + ["--variant", "fused_all_sam3"])
                    (run / "fused_other").mkdir()
                    with self.assertRaises(ValueError):
                        J.main(argv + ["--variant", "fused_other"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
