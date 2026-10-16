"""Synthetic regression tests. Never reads existing reconstruction runs or model weights."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "stereo"), str(CODE / "robot")]
os.environ.setdefault("V2D_ROOT", tempfile.gettempdir())
from motion_quality import (acceleration_loss, apply_symmetry, bundle_timing, debounce_phases,
                            interpolate_short_gaps, sticking_windows, spline_loss,
                            spline_projectors, spline_pose_prior)
from se3_fusion import fuse_trajectory
from fuse_objects import canon_measurements, carry_segments, damping_weights, grasp_segments, observed_positions, retime


def rot_err_deg(A, B):
    return np.rad2deg(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", A, B)).as_rotvec(), axis=1))


CUBE = np.array([[x, y, z] for x in (-.03, .03) for y in (-.03, .03) for z in (-.03, .03)])
import export_result_bundle as exporter


def loader_harness():
    """Exercise the actual loader boundary methods without MANO assets or robot imports."""
    source = ast.parse((CODE / "patches/ego_recon_loader.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "EgoReconDatasetLoader")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("load_mano_data", "get_fps")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    ego_fps = next(n.value.value for n in source.body if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "EGO_FPS" for t in n.targets))
    scope = dict(np=np, torch=torch, Rotation=Rotation, json=json, Path=Path, EGO_FPS=ego_fps)
    exec(compile(ast.fix_missing_locations(module), "loader_boundary", "exec"), scope)
    obj = scope[cls.name]()
    obj._bundle_fps = None  # __init__ is stripped with the toolkit base class
    obj._object_scale_cache = {}
    obj._scene_align = lambda _: (np.eye(3), np.zeros(3))
    obj._j0_template = lambda side, betas: np.zeros(3)
    obj._validate_world_repose = lambda *args: None
    return obj


class MotionTests(unittest.TestCase):
    def test_changed_python_sources_parse(self):
        for rel in ("stereo/motion_quality.py", "stereo/fuse_objects.py", "stereo/se3_fusion.py",
                    "stereo/symmetry.py", "stereo/s06h_grasp_plausible.py", "stereo/s06i_select_grasps.py",
                    "stereo/wrench_feasibility.py", "robot/export_result_bundle.py", "patches/ego_recon_loader.py"):
            ast.parse((CODE / rel).read_text(), filename=rel)

    def test_refinement_optimizer_and_fk_consistency(self):
        import s06h_grasp_plausible as refine
        import trimesh
        n = 16
        theta = torch.linspace(0, 2 * torch.pi, 778)
        template = torch.stack([.051 * torch.cos(theta), .051 * torch.sin(theta), theta * 0], -1)
        def forward(mano, go, hp, betas, is_right):
            R = refine.rotvec_to_matrix(go)
            v = torch.einsum("bij,nj->bni", R, template) + hp.mean(-1)[:, None, None] * .001
            return v, v[:, :21]
        class Sphere:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, p):
                return p.norm(dim=-1) - .05
        with tempfile.TemporaryDirectory() as tmp:
            root, run = Path(tmp), Path(tmp) / "runs/synthetic"
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write(run / "episode.json", {"fps": 20.})
            write(run / "objects_sam3/objects.json", [{"name": "sphere"}])
            write(run / "table_plane.json", {"up": [0, 0, 1], "height": -.1})
            write(run / "fusion_report.json", {"sphere": {"hand_contact": {"right": list(range(3, 13))}}})
            mesh = root / "meshes_obj/sphere/sphere.obj"
            mesh.parent.mkdir(parents=True)
            trimesh.creation.icosphere(subdivisions=1, radius=.05).export(mesh)
            for t in range(n):
                identity = {"R": np.eye(3).tolist(), "T": [0, 0, 0]}
                write(run / "recon_hawor/slam_poses" / f"{t:06d}.json", identity)
                write(run / "fused_all_sam3/sphere/poses" / f"{t:06d}.json", identity)
                shift = [(-1) ** t * .002, 0, 0]
                write(run / "stereo/mano_fit/right" / f"{t:06d}.json",
                      {"T": shift, "is_right": True, "joints": (template[:21] + torch.tensor(shift)).tolist(),
                       "mano": {"betas": [0] * 10, "global_orient": [0] * 3, "hand_pose": [0] * 45}})
            argv = ["refine", "synthetic", "--device", "cpu", "--iters", "30", "--out", "final"]
            with patch.object(refine, "ROOT", root), patch.object(refine, "Mano", lambda *args: None), \
                 patch.object(refine, "SDFGrid", Sphere), patch.object(refine, "mano_forward", forward), patch.object(sys, "argv", argv):
                refine.main()
            report = json.loads((run / "stereo/final/report.json").read_text())
            self.assertTrue(report["quality_passed"])
            for t in range(n):
                rec = json.loads((run / "stereo/final/right" / f"{t:06d}.json").read_text())
                self.assertLessEqual(np.linalg.norm(np.asarray(rec["T"]) - [(-1) ** t * .002, 0, 0]), .025)
                m = rec["mano"]
                v, j = forward(None, torch.tensor([m["global_orient"]]), torch.tensor([m["hand_pose"]]), None, True)
                np.testing.assert_allclose(j[0].numpy() + rec["T"], rec["joints"], atol=1e-6)
            # anchors are weighted per grasp segment and carry that segment's parameters
            anchors = report["sides"]["right"]["anchors"]
            self.assertTrue(anchors and all(r["grasp_start"] == 3 and r["squeeze_mm"] == 1.5 for r in anchors))
            self.assertIn("frames_excluded_deep", report["sides"]["right"]["quality"]["object"])
            # final pass: start from the selected hands, but trust limits are measured from mano_fit
            # (a 40 mm selected offset is pulled inside the 25 mm limit) and anchors reuse the
            # squeeze chosen for the grasp by s06i
            for t in range(n):
                rec = json.loads((run / "stereo/mano_fit/right" / f"{t:06d}.json").read_text())
                rec["T"] = (np.asarray(rec["T"]) + [.04 if t == 0 else .02, 0, 0]).tolist()
                write(run / "stereo/selected/right" / f"{t:06d}.json", rec)
            write(run / "stereo/selected/report.json", {
                "cands": ["final"], "default_params": {"squeeze_mm": 1.5, "contact_mm": 15.},
                "segments": [{"side": "right", "object": "sphere", "frames": [3, 12], "chosen": "cand",
                              "chosen_params": {"squeeze_mm": 3., "contact_mm": 15.}}]})
            argv = ["refine", "synthetic", "--device", "cpu", "--iters", "0", "--ref_hands", "mano_fit",
                    "--hands", "selected", "--segment_params", "selected", "--out", "phys"]
            with patch.object(refine, "ROOT", root), patch.object(refine, "Mano", lambda *args: None), \
                 patch.object(refine, "SDFGrid", Sphere), patch.object(refine, "mano_forward", forward), patch.object(sys, "argv", argv):
                refine.main()
            report = json.loads((run / "stereo/phys/report.json").read_text())
            self.assertTrue(all(r["squeeze_mm"] == 3. for r in report["sides"]["right"]["anchors"]))
            for t in range(n):
                ref = np.asarray(json.loads((run / "stereo/mano_fit/right" / f"{t:06d}.json").read_text())["T"])
                sel = np.asarray(json.loads((run / "stereo/selected/right" / f"{t:06d}.json").read_text())["T"])
                got = np.asarray(json.loads((run / "stereo/phys/right" / f"{t:06d}.json").read_text())["T"])
                if t == 0:
                    self.assertTrue(.0249 <= np.linalg.norm(got - ref) <= .025)
                else:
                    np.testing.assert_allclose(got, sel, atol=1e-5)

    def test_refinement_quality_statistic_is_robust(self):
        import s06h_grasp_plausible as refine
        from types import SimpleNamespace
        a = SimpleNamespace(pass_p95_mm=5., pass_frac_over_10mm=.05)
        self.assertTrue(refine.robust_stats(np.r_[np.zeros(99), 40.], a)["passed"])  # one bad frame
        s = refine.robust_stats(np.r_[np.zeros(90), np.full(10, 12.)], a)
        self.assertFalse(s["passed"])
        self.assertEqual(s["frames_over_10mm"], 10)
        self.assertTrue(refine.robust_stats([], a)["passed"])

    def test_export_timing_validity_and_variant_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs/synthetic"
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            n, fps = 14, 20.
            write(run / "episode.json", {"fps": fps})
            write(run / "objects_sam3/objects.json", [{"name": "cube"}])
            write(run / "stereo/mano_fit_phys/report.json", {"args": {"variant": "fused_all_sam3"}, "quality_passed": True})
            write(run / "table_plane.json", {"up": [0, 0, 1], "height": 0})
            write(run / "recon_hawor/intrinsics_stable.json", {"fx": 500, "fy": 500, "cx": 320, "cy": 240})
            mesh = root / "meshes_obj/cube/cube.obj"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            # A hand held still in the WORLD while the camera moves nonlinearly: a world-frame gap
            # fill reproduces it exactly, a camera-frame fill would not.
            Rw = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()
            pw, j0 = np.array([.1, .2, .4]), np.array([.01, -.02, .03])
            cams = {}
            # right: frames 3-5 missing (3 frames = 0.15 s: short), 11-13 trailing (held)
            # left: frame 0 leading (held), 6-9 missing (4 frames = 0.2 s: long, interpolated)
            missing = {"right": {3, 4, 5, 11, 12, 13}, "left": {0, 6, 7, 8, 9}}
            mir = {"right": np.ones(3), "left": np.array([1., -1., -1.])}  # left: mirrored-right params
            for t in range(n):
                frame = run / "recon_hawor/frames" / f"{t:06d}.png"
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.touch()
                C = Rotation.from_rotvec([.02 * t ** 1.5, .3 * np.sin(t), .01 * t]).as_matrix()
                c = np.array([.1 * np.sin(t), .02 * t ** 2, .01 * t])
                cams[t] = (C, c)
                write(run / "recon_hawor/slam_poses" / f"{t:06d}.json", {"R": C.tolist(), "T": c.tolist()})
                write(run / "fused_all_sam3/cube/poses" / f"{t:06d}.json", {"R": np.eye(3).tolist(), "T": [0, 0, .5]})
                T = C.T @ (pw - c) - j0
                for side in ("left", "right"):
                    if t in missing[side]:
                        continue
                    write(run / "stereo/mano_fit_phys" / side / f"{t:06d}.json",
                          {"T": T.tolist(), "joints": [(j0 + T).tolist()] * 21,
                           "mano": {"betas": [0] * 10, "global_orient": (Rotation.from_matrix(C.T @ Rw).as_rotvec() * mir[side]).tolist(),
                                    "hand_pose": [[0, 0, .05 * t]] * 15}})
            out = run / "robot/cube/outputs/result"
            with patch.object(exporter, "ROOT", root), patch.object(sys, "argv", ["export", "synthetic"]):
                exporter.main()
                manifest = json.loads((out / "manifest.json").read_text())
                with np.load(out / "result.npz") as data:
                    self.assertEqual(float(data["fps"]), fps)
                    np.testing.assert_allclose(data["timestamps"], np.arange(n) / fps)
                    for k in data.files:
                        if data[k].dtype.kind == "f":
                            self.assertTrue(np.isfinite(data[k]).all(), k)
                    # every frame is filled and valid; no zero/identity-at-camera hand anywhere
                    finger_truth = {"right": lambda t: .05 * min(t, 10), "left": lambda t: .05 * max(t, 1)}
                    for side in ("left", "right"):
                        self.assertTrue(data[f"hand_{side}_is_valid"].all())
                        np.testing.assert_array_equal(np.flatnonzero(data[f"hand_{side}_is_interpolated"]),
                                                      sorted(missing[side]))
                        self.assertTrue((np.abs(data[f"hand_{side}_wrist_to_camera_transform"][:, :3, 3]).sum(1) > 0).all())
                        for t in range(n):  # the hand stays where it is in the WORLD, in every frame
                            C, c = cams[t]
                            T = data[f"hand_{side}_wrist_trans_in_camera"][t]
                            o = data[f"hand_{side}_wrist_orient_in_camera"][t]
                            np.testing.assert_allclose(C @ (j0 + T) + c, pw, atol=1e-5, err_msg=f"{side} {t}")
                            np.testing.assert_allclose(C @ Rotation.from_rotvec(o * mir[side]).as_matrix(), Rw, atol=1e-5)
                            np.testing.assert_allclose(data[f"hand_{side}_finger_pose"][t],
                                                       [[0, 0, finger_truth[side](t)]] * 15, atol=1e-6)
                self.assertEqual(manifest["fps"], fps)
                np.testing.assert_allclose(manifest["timestamps"], np.arange(n) / fps)
                self.assertTrue(manifest["quality"]["passed"])
                g = manifest["hand_gaps"]
                self.assertEqual(g["max_missing_frames"], 3)
                self.assertEqual({k: g["right"][k] for k in ("n_interpolated_short", "n_interpolated_long",
                                                              "n_held_lead", "n_held_trail")}, dict(
                    n_interpolated_short=3, n_interpolated_long=0, n_held_lead=0, n_held_trail=3))
                self.assertEqual({k: g["left"][k] for k in ("n_interpolated_short", "n_interpolated_long",
                                                             "n_held_lead", "n_held_trail")}, dict(
                    n_interpolated_short=0, n_interpolated_long=4, n_held_lead=1, n_held_trail=0))
                # the exported bundle through the upstream loader boundary: bundle fps, finite world hands
                from types import SimpleNamespace
                loader = loader_harness()
                loader._j0_template = lambda side, betas: j0
                seq = SimpleNamespace(sequence_id="synthetic", source=SimpleNamespace(
                    npz_path=out / "result.npz", result_dir=out, object_scale=1.))
                got = loader.load_mano_data(seq, torch.device("cpu"))
                self.assertEqual(loader.get_fps(), fps)
                self.assertTrue(all(torch.isfinite(v).all() for v in got.values() if torch.is_tensor(v)))
                for side in ("left", "right"):  # world translation and orientation constant for both hands
                    np.testing.assert_allclose(got[f"{side}_trans"].numpy(), np.tile(pw - j0, (n, 1)), atol=1e-5)
                    go = got[f"{side}_global_orient"].numpy()
                    np.testing.assert_allclose(Rotation.from_rotvec(go).as_matrix(),
                                               np.tile(Rotation.from_rotvec(go[0]).as_matrix(), (n, 1, 1)), atol=1e-5)
                # --no_fill_long: only the short gap is filled; the rest are zeros, is_valid False
                with patch.object(sys, "argv", ["export", "synthetic", "--no_fill_long"]):
                    exporter.main()
                with np.load(out / "result.npz") as data:
                    np.testing.assert_array_equal(data["hand_right_is_valid"], [t < 11 for t in range(n)])
                    np.testing.assert_array_equal(data["hand_left_is_valid"], [t not in missing["left"] for t in range(n)])
                    np.testing.assert_array_equal(np.flatnonzero(data["hand_right_is_interpolated"]), [3, 4, 5])
                    for side in ("left", "right"):
                        bad = ~data[f"hand_{side}_is_valid"]
                        self.assertTrue((data[f"hand_{side}_wrist_trans_in_camera"][bad] == 0).all())
                        np.testing.assert_array_equal(data[f"hand_{side}_wrist_to_camera_transform"][bad],
                                                      np.tile(np.eye(4), (bad.sum(), 1, 1)))
                # a side without any observation: zeros, is_valid False, flagged in the manifest
                import shutil
                shutil.move(run / "stereo/mano_fit_phys/left", run / "left_hidden")
                exporter.main()
                shutil.move(run / "left_hidden", run / "stereo/mano_fit_phys/left")
                with np.load(out / "result.npz") as data:
                    self.assertFalse(data["hand_left_is_valid"].any())
                    self.assertTrue((data["hand_left_wrist_trans_in_camera"] == 0).all())
                    self.assertTrue(data["hand_right_is_valid"].all())
                self.assertTrue(json.loads((out / "manifest.json").read_text())["hand_gaps"]["left"]["no_observations"])
                write(run / "stereo/mano_fit_phys/report.json", {"args": {"variant": "fused_agent"}})
                with self.assertRaisesRegex(ValueError, "refined against"):
                    exporter.main()
                # a failed quality check exports loudly flagged; --strict refuses
                write(run / "stereo/mano_fit_phys/report.json", {"args": {"variant": "fused_all_sam3"}, "quality_passed": False})
                exporter.main()
                self.assertFalse(json.loads((out / "manifest.json").read_text())["quality"]["passed"])
                with patch.object(sys, "argv", ["export", "synthetic", "--strict"]), \
                        self.assertRaisesRegex(ValueError, "geometry failed"):
                    exporter.main()
                write(run / "stereo/mano_fit_phys/report.json", {"args": {"variant": "fused_all_sam3"}, "quality_passed": True})
                (run / "fused_all_sam3/cube/poses/000001.json").unlink()
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    exporter.main()

    def test_export_gap_boundaries(self):
        valid = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], bool)
        self.assertEqual(exporter.interior_gaps(valid, 3), [(2, 4)])
        self.assertEqual(exporter.interior_gaps(valid, 4), [(2, 4), (6, 9)])
        self.assertEqual(exporter.interior_gaps(np.zeros(4, bool), 3), [])

    def test_timing_requires_source_rate(self):
        with self.assertRaises(ValueError):
            bundle_timing({}, 10)
        self.assertEqual(bundle_timing({"fps": 20.25}, 10)[0], 20.25)
        with self.assertRaises(ValueError):
            bundle_timing({"fps": 30, "timestamps": np.arange(10) / 20}, 10)

    def test_short_gap_and_rotation_wrap(self):
        valid = np.array([True, False, True])
        angles = np.deg2rad([[0, 0, 179], [0, 0, 0], [0, 0, -179]])
        x, mask = interpolate_short_gaps(angles, valid, np.arange(3) / 20, rotations=True)
        self.assertAlmostEqual(abs(x[1, 2]), np.pi)
        np.testing.assert_array_equal(x[valid], angles[valid])
        self.assertTrue(mask[1])
        with self.assertRaises(ValueError):
            interpolate_short_gaps(angles, valid, np.arange(3) / 5)
        with self.assertRaises(ValueError):
            interpolate_short_gaps(angles, [False, True, True], np.arange(3) / 20)

    def test_loader_fps_from_bundle(self):
        """Patched upstream loader: get_fps = bundle fps (npz, else manifest), else EGO_FPS = 30."""
        from types import SimpleNamespace
        n = 4
        data = dict(camera_to_world_transform=np.tile(np.eye(4), (n, 1, 1)))
        for side in ("left", "right"):
            data.update({f"hand_{side}_betas": np.zeros(10), f"hand_{side}_finger_pose": np.zeros((n, 15, 3)),
                         f"hand_{side}_wrist_orient_in_camera": np.zeros((n, 3)),
                         f"hand_{side}_wrist_trans_in_camera": np.zeros((n, 3))})
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            seq = SimpleNamespace(sequence_id="s", source=SimpleNamespace(npz_path=d / "result.npz", result_dir=d,
                                                                          object_scale=1.))
            loader = loader_harness()
            self.assertEqual(loader.get_fps(), 30.)
            np.savez(d / "result.npz", fps=np.float64(20.), **data)
            loader.load_mano_data(seq, torch.device("cpu"))
            self.assertEqual(loader.get_fps(), 20.)
            np.savez(d / "result.npz", **data)  # no fps: must not inherit the previous sequence's
            loader.load_mano_data(seq, torch.device("cpu"))
            self.assertEqual(loader.get_fps(), 30.)
            (d / "manifest.json").write_text(json.dumps({"fps": 25.}))
            loader.load_mano_data(seq, torch.device("cpu"))
            self.assertEqual(loader.get_fps(), 25.)
            np.savez(d / "result.npz", fps=np.float64(np.nan), **data)
            with self.assertRaises(ValueError):
                loader.load_mano_data(seq, torch.device("cpu"))

    def test_contact_release_and_slip(self):
        p = np.zeros((12, 3))
        d = np.r_[np.zeros(5), np.ones(2) * .04, np.zeros(5)]
        windows = sticking_windows(np.arange(12), p, d)
        self.assertEqual(windows, [list(range(5)), list(range(7, 12))])
        p[:, 0] = np.arange(12) * .02
        self.assertEqual(sticking_windows(np.arange(12), p, np.zeros(12)), [])

    def test_cubic_spline_preserves_motion_and_contact_boundaries(self):
        n = 60
        t = np.arange(n) / 20
        poses = np.tile(np.eye(4), (n, 1, 1))
        truth = np.c_[.03 * t, .01 * t ** 2, .5 + t * 0]
        rng = np.random.default_rng(2)
        poses[:, :3, 3] = truth + rng.normal(0, .002, (n, 3))
        # Cross the axis-angle +/-pi boundary while the physical rotation stays smooth.
        angle = np.deg2rad(170 + 12 * t)
        poses[:, :3, :3] = Rotation.from_rotvec(np.c_[t * 0, t * 0, angle]).as_matrix()
        phases = ["grasp"] * 30 + ["released"] * 30
        smoothed = spline_pose_prior(poses, np.arange(n), 20., phases)
        self.assertLess(np.mean((smoothed[:, :3, 3] - truth) ** 2), np.mean((poses[:, :3, 3] - truth) ** 2))
        np.testing.assert_allclose(smoothed[:, :3, :3], poses[:, :3, :3], atol=1e-9)
        projectors = spline_projectors(np.arange(n), 20., phases)
        self.assertTrue(all(np.all(idx < 30) or np.all(idx >= 30) for idx, _ in projectors))
        x = torch.tensor(poses[:, :3, 3], requires_grad=True)
        loss = spline_loss(x, projectors)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_spline_does_not_bridge_occlusion(self):
        frames = np.r_[np.arange(10), np.arange(30, 40)]
        for idx, _ in spline_projectors(frames, 20.):
            self.assertTrue(np.all(idx < 10) or np.all(idx >= 10))

    def test_acceleration_uses_seconds_and_skips_gaps(self):
        losses = []
        for fps in (20., 40.):
            t = torch.arange(10, dtype=torch.float64) / fps
            x = torch.stack([.5 * t * t, 0 * t, 0 * t], -1)[:, None]
            losses.append(float(acceleration_loss(x, np.arange(10), fps)))
        self.assertAlmostEqual(losses[0], losses[1], places=12)
        x = torch.randn(3, 21, 3, requires_grad=True)
        loss = acceleration_loss(x, [0, 1, 30], 20.)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_smoothing_reduces_jitter_without_freezing_motion(self):
        torch.manual_seed(12)
        t = torch.arange(60, dtype=torch.float64) / 20
        truth = torch.stack([0.05 * t, .03 * torch.sin(t), 0.5 + 0 * t], -1)[:, None]
        noisy = truth + .002 * torch.randn_like(truth)
        x = noisy.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=.001)
        for _ in range(200):
            loss = ((x - noisy) ** 2).mean() + 5 * acceleration_loss(x, np.arange(60), 20.)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(torch.diff(x.detach(), n=2, dim=0).square().mean()),
                        float(torch.diff(noisy, n=2, dim=0).square().mean()) * .6)
        self.assertLess(float((x.detach() - truth).square().mean()), float((noisy - truth).square().mean()))
        self.assertGreater(float(x[-1, 0, 0] - x[0, 0, 0]), .12)

    def test_centroid_symmetry_preserves_geometry(self):
        c = np.array([.2, -.1, .3])
        vertices = c + np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        T = np.eye(4)
        T[:3, 3] = [.1, .3, .5]
        T[:3, :3] = Rotation.from_rotvec([.1, .2, .3]).as_matrix()
        G = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
        Q = apply_symmetry(T, G, c)
        a, b = vertices @ T[:3, :3].T + T[:3, 3], vertices @ Q[:3, :3].T + Q[:3, 3]
        self.assertLess(np.linalg.norm(a[:, None] - b[None], axis=-1).min(1).max(), 1e-10)

    def test_bimanual_grasp_segments(self):
        n = 8
        H = np.tile(np.eye(4), (n, 1, 1))
        hands = {s: {"H": H, "C": np.ones((n, 21))} for s in ("left", "right")}
        segs = grasp_segments(hands, [("left", "right")] * n, dict(enumerate(H)), n)
        self.assertEqual({g["side"] for g in segs}, {"left", "right"})
        self.assertAlmostEqual(sum(g["w"] for g in segs), 1.)

    def test_object_fusion_tracks_motion_and_reduces_noise(self):
        rng = np.random.default_rng(1)
        n = 24
        truth = np.tile(np.eye(4), (n, 1, 1))
        truth[:, 0, 3] = np.arange(n) * .003
        raw = truth.copy()
        raw[:, :3, 3] += rng.normal(0, .003, (n, 3))
        out, info = fuse_trajectory({"measurement": [(i, T, 1.) for i, T in enumerate(raw)]}, n,
                                    params={"fps": 20., "spline_weight": .2, "max_nfev": 60})
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(np.mean((out[:, :3, 3] - truth[:, :3, 3]) ** 2), np.mean((raw[:, :3, 3] - truth[:, :3, 3]) ** 2))
        self.assertGreater(out[-1, 0, 3] - out[0, 0, 3], .05)

    def test_resting_object_spinning_under_occlusion_is_damped(self):
        # A hand hovers (near, not holding) over a resting cube and hides it; FP spins it 6 deg/frame.
        rng = np.random.default_rng(3)
        n = 80
        truth = np.tile(np.eye(4), (n, 1, 1))
        truth[:, 2, 3] = .5
        near = np.zeros(n, bool)
        near[20:60] = True
        vis = np.ones(n)
        vis[25:55] = .1
        fp = []
        for t in range(n):
            M = truth[t].copy()
            if 25 <= t < 55:
                M[:3, :3] = Rotation.from_rotvec([0, 0, np.deg2rad(6 * (t - 24))]).as_matrix()
                M[:3, 3] += rng.normal(0, .004, 3)
            else:
                M[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.deg2rad(1), 3)).as_matrix()
                M[:3, 3] += rng.normal(0, .002, 3)
            fp.append((t, M, (vis[t], vis[t] ** 2)))
        plane = {"up": [0, 0, 1], "height": .47}
        static_w, _, rest = damping_weights(near, vis, np.zeros(n, bool), [], {t: T for t, T, _ in fp},
                                            20., CUBE, plane)
        self.assertGreater(static_w[26:54].min(), .8)       # occlusion damping is not gated by the rest test
        self.assertLess(rest[26:54].max(), .5)              # (which the spinning, hidden FP fails)
        out, _ = fuse_trajectory({"fp": fp}, n, params={"fps": 20., "static_w": static_w, "spline_weight": .2})
        free, _ = fuse_trajectory({"fp": fp}, n, params={"fps": 20., "spline_weight": .2})
        self.assertLess(rot_err_deg(out[25:55, :3, :3], truth[25:55, :3, :3]).mean(), 5.)
        self.assertGreater(rot_err_deg(free[25:55, :3, :3], truth[25:55, :3, :3]).mean(), 10.)
        # a lifted object held by a still hand keeps its held-still damping (no table support)
        lifted = {t: T @ np.diag([1., 1, 1, 1]) for t, T in enumerate(truth)}
        for T in lifted.values():
            T[2, 3] = .7
        H = {t: np.eye(4) for t in range(n)}
        held = np.ones(n, bool)
        sw, still, rest = damping_weights(np.ones(n, bool), np.ones(n) * .3, held,
                                          [{"frames": list(range(n)), "H": H}], lifted, 20., CUBE, plane)
        self.assertEqual(rest.max(), 0.)
        self.assertGreater(sw.min(), .99)

    def test_missing_table_plane_falls_back_to_ungated_damping(self):
        n = 30
        near = np.zeros(n, bool)
        near[10:20] = True
        vis = np.linspace(1, .2, n)
        held = np.zeros(n, bool)
        with self.assertWarns(UserWarning):
            sw, still, rest = damping_weights(near, vis, held, [], {}, 20., CUBE, None)
        free = (~near[:-1] & ~near[1:]).astype(float)
        occl = 1 - np.minimum(vis[:-1], vis[1:])
        np.testing.assert_allclose(sw, np.maximum(free, occl))
        np.testing.assert_array_equal(rest, 1.)

    def test_occluded_curved_carry_spline_is_not_biased_to_the_chord(self):
        # Carried on a 15 cm arc; mid-carry the FP weight falls below _init's 0.25 floor, so the
        # initial guess straight-lines the gap (a ~4 cm chord). The spline must smooth the solution,
        # not pull toward that guess.
        rng = np.random.default_rng(4)
        n = 60
        th = np.arange(n) / 20.
        truth = np.tile(np.eye(4), (n, 1, 1))
        truth[:, :3, 3] = np.c_[.15 * np.cos(th), .15 * np.sin(th), .5 + 0 * th]
        truth[:, :3, :3] = Rotation.from_rotvec(np.c_[0 * th, 0 * th, th]).as_matrix()
        fp = []
        for t in range(n):
            M = truth[t].copy()
            occ = 15 <= t < 45
            M[:3, 3] += rng.normal(0, .003, 3)
            M[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.deg2rad(15 if occ else 2), 3)).as_matrix() @ M[:3, :3]
            fp.append((t, M, (.1, .01) if occ else 1.))
        res = {}
        for w in (0., .2):
            P, info = fuse_trajectory({"fp": fp}, n, params={"fps": 20., "spline_weight": w})
            res[w] = (np.linalg.norm(P[15:45, :3, 3] - truth[15:45, :3, 3], axis=1).mean() * 1000,
                      rot_err_deg(P[15:45, :3, :3], truth[15:45, :3, :3]).mean())
            self.assertEqual(set(info["sources"]), {"fp"})  # no spline pseudo-measurements in stats
        self.assertLess(res[.2][0], 1.25 * res[0.][0] + 1.)   # mm
        self.assertLess(res[.2][1], 1.25 * res[0.][1] + 1.)   # deg
        self.assertLess(res[.2][0], 10.)                      # far from the 4 cm chord
        # spans shorter than the basis get no spline term at all (not an anchor to the init)
        _, info = fuse_trajectory({"fp": fp}, n, params={"fps": 20., "spline_weight": .2,
                                                         "contact_phases": [t // 3 for t in range(n)]})
        self.assertEqual(info["spline_spans"], 0)

    def test_fp_and_icp_start_on_different_symmetry_branches(self):
        G = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
        sym = {"group": [np.eye(3), G], "center": np.array([.01, -.02, .03])}
        n = 30
        truth = np.tile(np.eye(4), (n, 1, 1))
        truth[:, :3, :3] = Rotation.from_rotvec(np.c_[.1 + 0 * np.arange(n), 0 * np.arange(n), .02 * np.arange(n)]).as_matrix()
        truth[:, :3, 3] = [.1, .2, .5]
        flip = lambda T: apply_symmetry(T, G, sym["center"])
        fp = [(t, truth[t], (1., 1.)) for t in range(n)]
        icp = [(t, flip(truth[t]), 4.) for t in range(10, n)]            # ICP starts on the other branch
        glob = [(t, flip(truth[t]) if t % 2 else truth[t], 1.) for t in range(5, n)]
        out, ch = canon_measurements({"fp": fp, "icp": icp, "global": glob}, sym)
        self.assertEqual(ch, {"fp": 0, "icp": len(icp), "global": sum(t % 2 for t, _, _ in glob)})
        for src in out:
            for t, T, _ in out[src]:
                self.assertLess(rot_err_deg(T[None, :3, :3], truth[None, t, :3, :3])[0], 1e-6)
        alone, _ = canon_measurements(icp, sym)   # per-stream canonicalisation keeps ICP's own branch
        self.assertGreater(rot_err_deg(alone[0][1][None, :3, :3], truth[None, 10, :3, :3])[0], 170)

    def test_duplicate_icp_measurements_at_one_frame(self):
        G = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
        c = np.array([.01, -.02, .03])
        sym = {"group": [np.eye(3), G], "center": c}
        R = Rotation.from_rotvec([.1, .2, .3]).as_matrix()
        T0 = np.eye(4)
        T0[:3, :3], T0[:3, 3] = R, [.1, .2, .5]
        T1 = apply_symmetry(T0, G, c)
        T1[:3, 3] += [.003, 0, 0]
        icp = [(4, T0, 4.), (5, T0, 4.), (5, T1, 2.), (6, T0, 4.)]
        out, ch = canon_measurements({"fp": [], "icp": icp}, sym)
        self.assertEqual(len(out["icp"]), 4)
        self.assertEqual(ch["icp"], 1)
        for (t, Tin, w), (t2, Tout, w2) in zip(icp, out["icp"]):
            self.assertEqual((t, w), (t2, w2))
            g = Tin[:3, :3].T @ Tout[:3, :3]
            self.assertTrue(any(np.allclose(g, h, atol=1e-9) for h in sym["group"]))  # a group element
            self.assertLess(rot_err_deg(Tout[None, :3, :3], R[None])[0], 1e-6)
            np.testing.assert_allclose(Tout[:3, :3] @ c + Tout[:3, 3], Tin[:3, :3] @ c + Tin[:3, 3], atol=1e-12)

    def test_contact_phase_debounce(self):
        r = ("right",)
        phases = [r] * 10 + [()] + [r] * 10 + [()] * 2 + [r] + [()] * 10
        out = debounce_phases(phases, 3)
        self.assertEqual(out, [r] * 21 + [()] * 13)
        self.assertEqual(debounce_phases([()] * 5), [()] * 5)
        self.assertEqual(len(spline_projectors(np.arange(len(out)), 20., out)), 2)

    def test_spline_rotation_prior_beyond_150_degrees(self):
        rng = np.random.default_rng(5)
        n = 60
        truth = Rotation.from_rotvec(np.c_[np.zeros(n), np.zeros(n), np.deg2rad(np.linspace(0, 200, n))]).as_matrix()
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, np.deg2rad(2), (n, 3))).as_matrix() @ truth
        out = spline_pose_prior(poses, np.arange(n), 20.)
        self.assertLess(rot_err_deg(out[:, :3, :3], truth).mean(), .8 * rot_err_deg(poses[:, :3, :3], truth).mean())

    def test_spline_ridge_smooths_endpoints_and_keeps_linear_motion(self):
        frames = np.arange(40)
        (idx, P), = spline_projectors(frames, 20.)
        (_, P0), = spline_projectors(frames, 20., ridge=0.)
        lin = np.c_[frames * .01, 1 - frames * .02]
        np.testing.assert_allclose(P @ lin, lin, atol=1e-9)
        for end in (0, -1):   # endpoint noise gain
            self.assertLess(np.sum(P[end] ** 2), np.sum(P0[end] ** 2))
        self.assertEqual(spline_projectors(np.arange(6), 20., knot_seconds=.05), [])  # n_basis >= len: skipped


    def test_carry_starts_when_the_object_moves(self):
        # the palm touches at frame 10 and lifts from 10, but the object only leaves the table at 14
        n = 40
        P = np.zeros((n, 3))
        P[14:30, 2] = np.linspace(0.005, 0.2, 16)
        P[30:, :] = [0.3, 0, 0]          # put down elsewhere at 30
        P[20:22] = np.nan                 # occluded frames never decide
        H = {t: np.eye(4) for t in range(10, 34)}
        segs = [{"frames": list(range(10, 34)), "H": H, "side": "right", "w": 1.0}]
        (g,), log = carry_segments(segs, P, move=0.01, lead=1)
        self.assertEqual(g["frames"][0], 14)   # first frame > 1 cm from rest is 15; one frame of lead
        self.assertEqual(g["frames"][-1], 30)  # last frame > 1 cm from the put-down pose is 29; +1
        self.assertEqual(sorted(g["H"]), g["frames"])
        # consensus: from first contact (10) one source (ICP biased by the covering hand) reads 2 cm high
        Pb = P.copy(); Pb[10:14, 2] = 0.02
        (g2,), _ = carry_segments(segs, {"fp": P, "icp": Pb}, move=0.01, lead=1)
        self.assertEqual(g2["frames"][0], 14)
        # held but never moved: not carried
        segs2 = [{"frames": list(range(5, 20)), "H": {t: np.eye(4) for t in range(5, 20)}, "side": "left", "w": 1.0}]
        out, log2 = carry_segments(segs2, np.zeros((n, 3)), move=0.01)
        self.assertEqual(out, []); self.assertIsNone(log2[0]["carry"])
        # no observations at all: kept unchanged
        out3, _ = carry_segments(segs2, np.full((n, 3), np.nan))
        self.assertEqual(out3[0]["frames"], list(range(5, 20)))
        # observed_positions: weight-median over sources per frame
        W = lambda x: np.r_[np.c_[np.eye(3), [x, 0, 0]], [[0, 0, 0, 1]]]
        Q = observed_positions({"fp": [(0, W(0.0), (1.0, 1.0)), (1, W(0.0), (1.0, 1.0))],
                                "icp": [(0, W(1.0), 3.0)]}, 3, win=0)
        self.assertAlmostEqual(Q[0, 0], 1.0); self.assertTrue(np.isnan(Q[2, 0]))

    def test_retime_moves_stereo_measurements_to_colour_time(self):
        # the stereo stream leads the video by 0.75 frame: ICP at frame k shows the object at k + 0.75
        v = 0.01  # m per frame
        slam = {t: np.eye(4) for t in range(20)}
        W = lambda x, ang: np.r_[np.c_[Rotation.from_rotvec([0, 0, ang]).as_matrix(), [x, 0, 0]], [[0, 0, 0, 1]]]
        icp = [(k, W(v * (k + 0.75), 0.02 * (k + 0.75)), 2.0 + k) for k in range(20)]
        out = retime(icp, 0.75, slam)
        by = {t: (T, w) for t, T, w in out}
        for t in range(1, 20):
            np.testing.assert_allclose(by[t][0][:3, 3], [v * t, 0, 0], atol=1e-12)
            self.assertAlmostEqual(Rotation.from_matrix(by[t][0][:3, :3]).as_rotvec()[2], 0.02 * t, places=9)
            self.assertEqual(by[t][1], 2.0 + (t - 1))  # the smaller neighbour weight
        self.assertEqual(by[0][1], 1.0)                  # frame 0: nearest measurement (k = 0) at half weight
        np.testing.assert_allclose(by[0][0][:3, 3], [v * 0.75, 0, 0], atol=1e-12)
        self.assertIs(retime(icp, 0.0, slam), icp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
