"""Synthetic tests of robot/rest_settle.py: a tall narrow box (2 x 5 x 14 cm, like the mini sweeper) resting
slightly tilted and sunk into the table, then lifted."""
from pathlib import Path
import sys
import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot"))
import rest_settle as RS  # noqa: E402

BOX = trimesh.creation.box(extents=[0.02, 0.05, 0.14])
V, COM = np.asarray(BOX.vertices), np.zeros(3)


def pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.as_matrix(), p
    return T


def zmin(T):
    return (V @ T[:3, :3].T)[:, 2].min() + T[2, 3]


def traj(tilt_deg, sink=0.003, n_rest=30, n_lift=20, yaw_deg=30.0):
    R = Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("x", tilt_deg, degrees=True)
    T0 = pose(R, [0.05, -0.02, 0.0])
    T0[2, 3] -= zmin(T0) + sink                                      # lowest point `sink` below the table
    out = [T0.copy() for _ in range(n_rest)]
    for k in range(1, n_lift + 1):                                    # lifted straight up at 10 cm/s
        T = T0.copy()
        T[2, 3] += 0.005 * k
        out.append(T)
    return np.array(out)


class RestSettleTest(unittest.TestCase):
    def test_tilted_sunk_box_is_settled_flat_on_the_table(self):
        o2w = traj(2.5)
        out, rep = RS.settle(o2w, V, COM)
        self.assertEqual(len(rep), 1)
        self.assertTrue(rep[0]["applied"])
        a, b = rep[0]["frames"]
        self.assertEqual(a, 0)
        for t in range(a, b):
            up = out[t, :3, :3] @ np.array([0, 0, 1.0])
            self.assertLess(np.degrees(np.arccos(min(1.0, abs(up[2])))), 0.05)   # flat
            self.assertAlmostEqual(zmin(out[t]), 0.0, places=6)         # bottom on the table
            np.testing.assert_allclose(out[t, :2, 3], o2w[t, :2, 3], atol=1e-6)  # COM xy kept (COM = origin)
            yaw = lambda T: np.degrees(np.arctan2(T[1, 0], T[0, 0]))
            self.assertAlmostEqual(yaw(out[t]), yaw(o2w[t]), delta=0.2)  # yaw kept
        # the lift beyond the ramp is untouched, and the transition is continuous
        np.testing.assert_allclose(out[b + RS.DEFAULTS["ramp"]:], o2w[b + RS.DEFAULTS["ramp"]:])
        steps = np.linalg.norm(np.diff(out[:, :3, 3], axis=0), axis=1)
        self.assertLess(steps.max(), 0.0055)

    def test_stable_face_margin_of_the_base(self):
        faces = RS.stable_faces(V, COM)
        base = [m for n, m in faces if n[2] < -0.99]
        self.assertEqual(len(base), 1)
        self.assertAlmostEqual(base[0], 0.01, places=4)                 # half the 2 cm width

    def test_leaning_object_is_left_alone(self):
        o2w = traj(20.0, sink=0.0)
        out, rep = RS.settle(o2w, V, COM)
        self.assertTrue(rep and not any(r["applied"] for r in rep))
        np.testing.assert_allclose(out, o2w)

    def test_object_off_the_table_disk_is_left_alone(self):
        o2w = traj(2.5)
        out, rep = RS.settle(o2w, V, COM, disk={"center": [1.0, 1.0, 0.0], "radius": 0.2})
        self.assertEqual(rep, [])
        np.testing.assert_allclose(out, o2w)

    def test_held_object_in_the_air_is_left_alone(self):
        o2w = traj(2.5)
        o2w[:, 2, 3] += 0.05
        out, rep = RS.settle(o2w, V, COM)
        self.assertEqual(rep, [])

    def test_short_gap_between_rest_segments_keeps_the_correction(self):
        o2w = traj(2.5, n_lift=0, n_rest=40)
        o2w[18:21, 0, 3] += np.array([0.01, 0.02, 0.03])           # a 3-frame nudge splits the rest in two
        out, rep = RS.settle(o2w, V, COM)
        self.assertEqual(sum(r["applied"] for r in rep), 2)
        for t in range(len(out)):
            self.assertAlmostEqual(zmin(out[t]), 0.0, places=5)    # never sinks in the gap


if __name__ == "__main__":
    unittest.main(verbosity=2)
