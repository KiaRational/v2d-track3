"""Synthetic tests for the refinement agent's critic plumbing and the mocap evaluation.
Never reads reconstruction runs or model weights (the kit's AUC.py is loaded for a cross-check)."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "refine_agent"), str(CODE / "stages")]
os.environ.setdefault("V2D_ROOT", tempfile.gettempdir())
os.environ.setdefault("V2D_DATA", tempfile.gettempdir())
import agent  # noqa: E402
import validate_critic  # noqa: E402
import verify  # noqa: E402
from eval_objects import kit_add_auc, kit_style_metrics, robust_align  # noqa: E402

KIT_AUC = Path(os.environ.get("V2D_KIT", "/scratch/kghasemz/v2d/submission_kit/v2d_submission_kit")) \
    / "metric_code/track_3/AUC.py"


def pose(R, p):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


def rot_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def trajectories(rng, n=60, objs=2):
    """World poses (n, 4, 4) per object: slow random motion."""
    out = []
    for _ in range(objs):
        rv = np.cumsum(rng.normal(0, 0.03, (n, 3)), 0) + rng.normal(0, 1, 3)
        p = np.cumsum(rng.normal(0, 0.01, (n, 3)), 0) + rng.normal(0, 0.3, 3)
        out.append(np.stack([pose(Rotation.from_rotvec(r).as_matrix(), q) for r, q in zip(rv, p)]))
    return out


class NearHand(unittest.TestCase):
    def setUp(self):
        d = np.random.default_rng(0).normal(size=(2000, 3))
        self.tree = cKDTree(0.05 * d / np.linalg.norm(d, axis=1, keepdims=True))  # 5 cm sphere, object frame
        self.Wc = pose(Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(), [0.1, 0, 0.5])

    def test_contacts_padded_any_side(self):
        c = verify.contact_frames({"hand_contact": {"left": [10], "right": [30, 31]}})
        self.assertEqual(c, set(range(8, 13)) | set(range(28, 34)))
        self.assertEqual(verify.contact_frames(None), set())
        self.assertTrue(verify.hand_near_joints(12, self.Wc, self.tree, {}, c))
        self.assertFalse(verify.hand_near_joints(13, self.Wc, self.tree, {}, c))

    def test_joint_distance(self):
        out = self.Wc[:3, :3] @ np.array([1.0, 0, 0])  # object-frame +x direction, in camera frame
        near = np.tile(self.Wc[:3, 3] + 0.065 * out, (21, 1))   # 1.5 cm off the surface
        far = np.tile(self.Wc[:3, 3] + 0.11 * out, (21, 1))     # 6 cm off
        far_nan = far.copy()
        far_nan[0] = np.nan
        self.assertTrue(verify.hand_near_joints(5, self.Wc, self.tree, {"right": {5: near.tolist()}}))
        self.assertFalse(verify.hand_near_joints(5, self.Wc, self.tree, {"right": {5: far.tolist()}, "left": {}}))
        self.assertFalse(verify.hand_near_joints(5, self.Wc, self.tree, {"right": {5: far_nan}}))
        self.assertFalse(verify.hand_near_joints(6, self.Wc, self.tree, {"right": {5: near.tolist()}}))


class MissingFramePenalty(unittest.TestCase):
    def test_missing_counts_as_error(self):
        base = {t: 0.3 for t in range(4)}
        alone = {t: 0 for t in range(4)}
        frames = sorted(base)
        self.assertAlmostEqual(agent.objective(base, alone, frames=frames), 0.3)
        # the trial pushes two frames off-image (features None) / loses one pose: must not look better
        trial = {0: 0.1, 1: 0.1, 2: None}
        J = agent.objective(trial, {0: 0, 1: 0, 2: 0}, frames=frames)
        self.assertAlmostEqual(J, (0.1 + 0.1 + 1 + 1) / 4)
        self.assertGreater(J, agent.objective(base, alone, frames=frames))
        self.assertEqual(agent.n_missing(trial, frames=frames), 2)
        self.assertEqual(agent.n_missing(trial, 0, 1, frames), 0)
        self.assertAlmostEqual(agent.objective(trial, {}, 0, 1, frames), 0.1)
        self.assertIsNone(agent.objective({}, {}, 5, 9, frames))
        self.assertEqual(agent.fmt(None), "n/a")

    def test_loeo_calibration_selected(self):
        loeo = {"features": agent.CAL["features"], "transform": agent.CAL["transform"],
                "episodes": {"public_ep000000": {"coef": [0.0, 0.0, 0.0], "intercept": 0.0}}}
        with patch.object(agent, "CAL_LOEO", loeo):
            c = agent.calibration("public_ep000000")
            self.assertEqual(c["intercept"], 0.0)
            ser = {f: {"3": 0.5} for f in c["features"]}
            self.assertAlmostEqual(agent.err_prob(ser, 3, c), 0.5)
            self.assertIs(agent.calibration("evaluation_ep000001"), agent.CAL)

    def test_window_overlap(self):
        self.assertTrue(agent.overlaps((5, 8), [(0, 5)]))
        self.assertFalse(agent.overlaps((6, 8), [(0, 5), (9, 12)]))


class SE3Alignment(unittest.TestCase):
    def test_recovers_known_transform_with_noise_and_outliers(self):
        rng = np.random.default_rng(1)
        G = trajectories(rng, 80, 3)
        Rk = Rotation.from_euler("zyx", [70, -20, 15], degrees=True).as_matrix()
        tk = np.array([1.0, -2.0, 0.5])
        E, Gp = [], []
        for obj in G:
            for i, M in enumerate(obj):
                e = np.linalg.inv(pose(Rk, tk)) @ M              # our world = mocap mapped back
                e[:3, 3] += rng.normal(0, 0.003, 3)
                e[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(1), 3)).as_matrix() @ e[:3, :3]
                if rng.random() < 0.1:                           # a lost track: wrong place, wrong way up
                    e = pose(Rotation.random(random_state=i).as_matrix(), e[:3, 3] + rng.normal(0, 0.3, 3))
                E.append(e)
                Gp.append(M)
        s, R, t, keep = robust_align(E, Gp)
        self.assertEqual(s, 1.0)
        self.assertLess(rot_deg(R, Rk), 0.3)
        self.assertLess(np.linalg.norm(t - tk), 0.003)
        self.assertGreater(keep.mean(), 0.8)
        s3, R3, t3, _ = robust_align(E, Gp, with_scale=True)
        self.assertLess(abs(s3 - 1), 0.02)
        # a symmetric object (continuous about z) whose spin is wrong everywhere must not tilt the frame
        spin = [e @ pose(Rotation.from_euler("z", 90, degrees=True).as_matrix(), [0, 0, 0]) for e in E]
        s, R, t, _ = robust_align(spin, Gp, syms=[("continuous", [0, 0, 1])] * len(E))
        self.assertLess(rot_deg(R, Rk), 0.3)


class KitStyle(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.G = trajectories(rng, 50, 2)
        self.V = [rng.normal(0, 0.04, (300, 3)) for _ in range(2)]
        A = pose(Rotation.from_euler("xyz", [30, 40, -50], degrees=True).as_matrix(), [3, -1, 2])
        self.E = [np.einsum("ij,njk->nik", A, g) for g in self.G]      # perfect up to one rigid transform

    def test_perfect_and_offset(self):
        m = kit_style_metrics(self.E, self.G, self.V)
        self.assertAlmostEqual(m["AUC"], 1.0)
        self.assertTrue(m["SP_SR"] and m["MP_SR"])
        self.assertEqual((m["align_frame"], m["align_fallback"], m["missing_frames"]), (0, False, 0))
        E = [self.E[0], self.E[1].copy()]
        E[1][:, :3, 3] += self.E[1][:, :3, :3] @ np.array([0.045, 0, 0])   # slot 1 always 4.5 cm off
        m = kit_style_metrics(E, self.G, self.V)
        self.assertAlmostEqual(m["per_object"][1]["AUC"], 4.5 / 8)       # ADD < t for t = 5..9 cm
        self.assertAlmostEqual(m["AUC"], (1 + 4.5 / 8) / 2)
        self.assertTrue(m["SP_SR"])
        self.assertFalse(m["MP_SR"])

    def test_missing_frames_fail(self):
        E = [self.E[0].copy(), self.E[1].copy()]
        E[1][::2] = np.nan
        m = kit_style_metrics(E, self.G, self.V)
        self.assertEqual(m["missing_frames"], 25)
        self.assertAlmostEqual(m["per_object"][1]["AUC"], 0.5)
        self.assertFalse(m["SP_SR"])
        self.assertTrue(m["SP_SR_available"])
        E[0][:3] = np.nan                                                 # slot 0 missing at its first frame
        m = kit_style_metrics(E, self.G, self.V)
        self.assertEqual((m["align_frame"], m["align_fallback"]), (3, True))
        self.assertAlmostEqual(kit_add_auc([np.inf] * 4), 0.0)

    def test_matches_kit_code(self):
        try:
            import pandas  # noqa: F401
        except ImportError:  # the scorer core is pure numpy; pandas only feeds its I/O and annotations
            import types
            pd_ = types.ModuleType("pandas")
            pd_.DataFrame = object
            pd_.api = types.ModuleType("pandas.api")
            pd_.api.types = types.ModuleType("pandas.api.types")
            sys.modules.update({"pandas": pd_, "pandas.api": pd_.api, "pandas.api.types": pd_.api.types})
        try:
            spec = importlib.util.spec_from_file_location("kit_auc", KIT_AUC)
            kit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kit)
        except (ImportError, FileNotFoundError, OSError) as e:
            self.skipTest(f"kit AUC.py not importable here: {e}")
        rng = np.random.default_rng(3)
        for trial in range(5):
            E = []
            for g in self.E:
                e = g.copy()
                e[:, :3, 3] += rng.normal(0, 0.01 * (trial + 1), (len(e), 3))
                e[:, :3, :3] = np.einsum("nij,njk->nik", Rotation.from_rotvec(
                    rng.normal(0, 0.05 * (trial + 1), (len(e), 3))).as_matrix(), e[:, :3, :3])
                E.append(e)
            to7 = lambda Ms: np.stack([np.c_[M[:, :3, 3], Rotation.from_matrix(M[:, :3, :3]).as_quat()] for M in Ms], 1)[:, None]
            ach, ref = to7(E), to7(self.G)                                # (T, 1 world, B, 7), xyzw
            warped = kit._warp_to_reference_start(ach, ref)
            ref_auc = kit._add_auc(kit._object_add(warped, ref, np.stack(self.V)))
            m = kit_style_metrics(E, self.G, self.V)
            self.assertAlmostEqual(m["AUC"], ref_auc, places=9)
            self.assertEqual(m["SP_SR"], bool(kit._spider_success(warped, ref)[0]))
            self.assertEqual(m["MP_SR"], bool(kit._maniptrans_success(warped, ref, np.arange(2))[0]))


class TieAwareAUC(unittest.TestCase):
    def test_ties_count_half(self):
        self.assertAlmostEqual(validate_critic.auc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)
        self.assertAlmostEqual(validate_critic.auc([0, 0, 1, 1], [0, 1, 0, 1]), 0.5)
        self.assertAlmostEqual(validate_critic.auc([0, 1, 1, 2], [0, 0, 1, 1]), 0.875)
        rng = np.random.default_rng(4)
        s, y = rng.integers(0, 3, 200), rng.random(200) < 0.3
        pos, neg = s[y], s[~y]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        self.assertAlmostEqual(validate_critic.auc(s, y), brute)
        self.assertIsNone(validate_critic.auc([1, 2], [1, 1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
