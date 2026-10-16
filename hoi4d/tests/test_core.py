"""README engineering requirements: coordinate transforms, time derivatives, symmetry comparisons, cavity
collisions, failure propagation. Run: PYTHONPATH=hoi4d/src python hoi4d/tests/test_core.py (agile image; pytest also works)."""
import numpy as np
from scipy.spatial.transform import Rotation

from hoi4d import io as hio
from hoi4d.diagnostics import motion as hm
from hoi4d.geometry.assets import inv, transform
from hoi4d.ingest.manifest import world_alignment


def test_transform_roundtrip():
    rng = np.random.default_rng(0)
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=1).as_matrix()
    T[:3, 3] = rng.normal(size=3)
    p = rng.normal(size=(10, 3))
    assert np.allclose(transform(inv(T), transform(T, p)), p)
    assert np.allclose(inv(T) @ T, np.eye(4))


def test_world_alignment_puts_table_at_zero():
    up = np.array([-0.1, -0.75, -0.65]); up /= np.linalg.norm(up)
    centre = np.array([0.06, 0.2, 0.55])
    h = float(centre @ up)
    A = world_alignment({"up": up.tolist(), "height": h, "center": centre.tolist()})
    q = centre + 0.1 * np.cross(up, [1, 0, 0])  # another point on the table plane
    assert abs(transform(A, centre[None])[0, 2]) < 1e-9 and abs(transform(A, q[None])[0, 2]) < 1e-9
    assert np.allclose(transform(A, centre[None])[0, :2], 0, atol=1e-9)
    assert np.allclose(A[:3, :3] @ up, [0, 0, 1])


def test_derivatives_use_real_timestamps():
    t = np.cumsum(np.r_[0, np.full(49, 1 / 20.25)])
    x = (0.5 * 3.0 * t ** 2)[:, None]  # a = 3 m/s^2
    a = hm.finite_diff(x, t, 2)
    assert np.allclose(a[3:-3], 3.0, atol=1e-6)


def test_symmetric_rotation_is_not_a_jump():
    ax = np.array([0, 0, 1.0])
    sym = [Rotation.from_rotvec(ax * a).as_matrix() for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
    Ra = np.eye(3)[None]
    Rb = Rotation.from_rotvec(ax * np.deg2rad(90)).as_matrix()[None]
    assert hm.geodesic(Ra, Rb)[0] > 1.5
    assert hm.geodesic(Ra, Rb, sym)[0] < 1e-6


def test_constant_velocity_motion_has_no_jump_residual():
    F = 30
    t = np.arange(F) / 20.0
    T = np.tile(np.eye(4), (F, 1, 1))
    T[:, 0, 3] = 2.0 * t  # fast but smooth: 2 m/s
    m = hm.object_motion(T, np.ones(F, bool), t)
    assert np.nanmax(m["jump_resid_m"]) < 1e-9
    T[15, 0, 3] += 0.05
    m = hm.object_motion(T, np.ones(F, bool), t)
    assert np.nanmax(m["jump_resid_m"]) > 0.04


def test_missing_frames_are_nan_never_identity():
    tr = hio.Trajectories(np.arange(5) / 20.0, np.arange(5))
    T = np.tile(np.eye(4), (5, 1, 1))
    valid = np.array([1, 1, 0, 1, 1], bool)
    tr.add_object("x", T, valid, np.where(valid, 1, 0))
    assert np.isnan(tr.objects["x"]["T_world_obj"][2]).all()
    tr.objects["x"]["T_world_obj"][2] = np.eye(4)  # an identity placeholder must be rejected
    try:
        hio.validate_trajectories(tr)
    except AssertionError:
        return
    raise AssertionError("identity placeholder was accepted")


def test_cavity_collision():
    """A point inside a cup's cavity is not penetrating; a point inside its wall is."""
    from hoi4d.geometry.assets import ObjectAsset
    a = ObjectAsset.__new__(ObjectAsset)
    # wall = thin box x in [0.04, 0.045]; cavity = x < 0.04 is free space
    from scipy.spatial import ConvexHull
    v = np.array([[x, y, z] for x in (0.04, 0.045) for y in (-0.05, 0.05) for z in (0, 0.1)])
    h = ConvexHull(v)
    a.parts = [(h.equations[:, :3], h.equations[:, 3])]
    pen = a.penetration(np.array([[0.0, 0, 0.05], [0.0425, 0, 0.05]]))
    assert pen[0] == 0 and pen[1] > 0.002


if __name__ == "__main__":
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", k)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", k, repr(e))
    raise SystemExit(fails)
