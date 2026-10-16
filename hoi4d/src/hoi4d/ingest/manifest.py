"""README stages 1 + 5/6 (adapter): scene_manifest.json + trajectories_initial.npz from the existing pipeline.

Existing layout (see GAP_MAP.md): reference camera = undistorted ego_cam_a (intrinsics_stable.json); poses
are camera-from-object per colour frame (fused_all_sam3/<obj>/poses/NNNNNN.json {R,T}); camera-to-world =
DROID-SLAM (recon_hawor/slam_poses, quaternion wxyz); table_plane.json in SLAM world.
hoi4d world: W = A @ slam, A rotates the table normal to +z and puts the table top at z=0, origin at the
table-disk centre (same construction as robot/export_result_bundle.py, plus the xy centring).

Hands: MANO parameters stay in the pipeline's per-frame camera frame (left hands use the pipeline's mirrored
right-hand parameterisation: verts = mirror_x(MANO_right(params)) + T); joints are converted to the world frame.
T_world_cam (F,4,4) is stored so parameters can be mapped exactly when needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .. import conventions as C
from .. import io as hio

ROOT = hio.ROOT


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def slam_poses(run: Path, n: int) -> np.ndarray:
    out = np.full((n, 4, 4), np.nan)
    for i in range(n):
        p = run / "recon_hawor/slam_poses" / f"{i:06d}.json"
        if p.exists():
            d = json.loads(p.read_text())
            q = d["rotation"]
            out[i] = _T(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), d["translation"])
    return out


def world_alignment(tp: dict) -> np.ndarray:
    """A: SLAM world -> hoi4d world (table normal = +z, table top z = 0, disk centre at x = y = 0)."""
    up = np.asarray(tp["up"], float)
    up /= np.linalg.norm(up)
    R = Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()
    A = _T(R, np.zeros(3))
    c = A[:3, :3] @ np.asarray(tp.get("center", up * tp["height"]), float)
    A[:3, 3] = -np.array([c[0], c[1], tp["height"]])
    # height: p.up = h on the table -> after rotation z = h; shift to 0
    return A


def entity_registry(run: Path, objects: list[str]) -> list[dict]:
    ents = []
    oj = run / "objects_sam3/objects.json"
    ids = {o["name"]: o["id"] for o in json.loads(oj.read_text())} if oj.exists() else {}
    for n in objects:
        ents.append({"id": n, "kind": "rigid_object", "label": n.replace("_", " "), "mask_id": ids.get(n),
                     "mesh": str(ROOT / "meshes_obj" / n / f"{n}.obj"), "views": ["ego_cam_a"],
                     "stereo_association": "depth reprojected into ego_cam_a (pixel-aligned); no separate b/c masks"})
    for side, mid in (("right", 3), ("left", 2)):
        ents.append({"id": f"hand_{side}", "kind": "hand", "label": f"{side} hand", "mask_id": mid,
                     "views": ["ego_cam_a", "ego_cam_b", "ego_cam_c"]})
    ents.append({"id": "table", "kind": "support", "label": "round table", "views": ["ego_cam_a"]})
    return ents


def build_manifest(run_name: str, cfg: dict) -> tuple[dict, np.ndarray]:
    run = ROOT / "runs" / run_name
    ep = json.loads((run / "episode.json").read_text())
    n = len(sorted((run / "recon_hawor/frames").glob("*.png")))
    fps = float(ep["fps"])
    t_s = np.arange(n) / fps
    tp = json.loads((run / "table_plane.json").read_text())
    A = world_alignment(tp)
    slam = slam_poses(run, n)
    T_wc = np.einsum("ij,fjk->fik", A, slam)
    K = json.loads((run / "intrinsics_stable.json").read_text())
    calib = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
    objects = ep.get("objects") or [ep.get("primary_object")]
    objects = [o for o in objects if o and (run / cfg["inputs"]["objects_variant"] / o).is_dir()]
    man = {
        "run": run_name, "split": ep.get("split"), "task_label": ep.get("task"),
        "units": C.UNITS, "quaternion_order": C.QUAT_ORDER, "transform_convention": C.TRANSFORM_CONVENTION,
        "world_frame": {"definition": "z up (table normal, opposite gravity), origin at the table-disk centre on the "
                                      "table top; = A @ DROID-SLAM world", "A_world_slam": A,
                        "gravity_source": "table_plane.json (stereo RANSAC plane normal; table assumed level)",
                        "table_vs_mocap_vertical_deg": tp.get("gt_angle_deg")},
        "gravity_world": [0.0, 0.0, -9.81],
        "fps": fps, "frame_count": n,
        "timestamps": {"t_s": t_s, "source": "frame_index / fps (episode.json has no per-frame timestamps)"},
        "cameras": {"reference": "ego_cam_a (undistorted colour)",
                    "ego_cam_a": {"K": [[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]],
                                  "size": [K["width"], K["height"]], "distortion": "none (undistorted frames)"},
                    "stereo": {k: calib["stereo"][k] for k in ("left", "right", "baseline_m")},
                    "T_cam_c_cam_a": calib["T_ego_cam_a_to_ego_cam_c"],
                    "camera_to_world": "T_world_cam[f] = A_world_slam @ slam_poses[f] (stored in trajectories npz)",
                    "rig": "head-mounted, moving; calibration verified (s01, stereo epipolar err 1.9 px)"},
        "support": {"kind": "disk", "radius_m": tp.get("radius"), "thickness_m": tp.get("thickness", 0.03),
                    "top_z_m": 0.0, "fit": tp.get("disk_fit")},
        "entities": entity_registry(run, objects),
        "scale_sources": {"depth": "FoundationStereo disparity, Z = f*B/d, B = calib stereo baseline",
                          "objects": "scanned meshes (metric) / SAM 3D mesh rescaled by stereo (wooden_spoon)",
                          "hands": "MANO fitted to stereo triangulated joints"},
        "inputs": {"objects_variant": cfg["inputs"]["objects_variant"]},
        "validity": {"frames_dir": str(run / "recon_hawor/frames"), "depth_dir": str(run / "depth_stereo"),
                     "depth_coverage": _depth_cov(run)},
        "created": hio.now(),
    }
    return hio.validate("scene_manifest", man), T_wc


def _depth_cov(run):
    p = run / "depth_stereo_stats.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"mean": d.get("mean_coverage")}


def pick_hands(run: Path, cfg: dict) -> str | None:
    for h in [cfg["inputs"]["hands_variant"], *cfg["inputs"]["hands_fallbacks"]]:
        if (run / "stereo" / h / "right").is_dir():
            return h
    return None


def build_initial(run_name: str, cfg: dict, man: dict, T_wc: np.ndarray) -> hio.Trajectories:
    run = ROOT / "runs" / run_name
    n = man["frame_count"]
    t_s = np.asarray(man["timestamps"]["t_s"])
    var = cfg["inputs"]["objects_variant"]
    tr = hio.Trajectories(t_s, np.arange(n), {"run": run_name, "revision": "initial", "parent": None,
                                              "created": hio.now(), "inputs": {"objects": var}})
    camvalid = np.isfinite(T_wc).all(axis=(1, 2))
    for e in man["entities"]:
        if e["kind"] != "rigid_object":
            continue
        nm = e["id"]
        T = np.full((n, 4, 4), np.nan)
        src = np.zeros(n, np.uint8)
        for i in range(n):
            p = run / var / nm / "poses" / f"{i:06d}.json"
            if not p.exists() or not camvalid[i]:
                continue
            d = json.loads(p.read_text())
            T[i] = T_wc[i] @ _T(np.asarray(d["R"]), d["T"])
            has_fp = (run / "objects_sam3" / nm / "poses" / f"{i:06d}.json").exists()
            has_icp = (run / "objects_sam3" / nm / "stereo_icp" / f"{i:06d}.json").exists()
            src[i] = C.SOURCE_IDS["fused"] if (has_fp or has_icp) else C.SOURCE_IDS["interpolated"]
        valid = np.isfinite(T).all(axis=(1, 2))
        # re-orthonormalise (JSON round-off)
        for i in np.where(valid)[0]:
            U, _, Vt = np.linalg.svd(T[i, :3, :3])
            T[i, :3, :3] = U @ Vt
        tr.add_object(nm, T, valid, src)
    hv = pick_hands(run, cfg)
    tr.meta["inputs"]["hands"] = hv
    tr.meta["T_world_cam_valid"] = camvalid.tolist()
    if hv:
        for side in ("right", "left"):
            go = np.full((n, 3), np.nan); hp = np.full((n, 45), np.nan); tl = np.full((n, 3), np.nan)
            J = np.full((n, 21, 3), np.nan); valid = np.zeros(n, bool); src = np.zeros(n, np.uint8); betas = None
            for i in range(n):
                p = run / "stereo" / hv / side / f"{i:06d}.json"
                if not p.exists() or not camvalid[i]:
                    continue
                d = json.loads(p.read_text())
                betas = d["mano"]["betas"]
                go[i] = d["mano"]["global_orient"]; hp[i] = d["mano"]["hand_pose"]; tl[i] = d["T"]
                Jc = np.asarray(d["joints"], float)
                J[i] = Jc @ T_wc[i, :3, :3].T + T_wc[i, :3, 3]
                valid[i] = True
                obs = d.get("has_triangulation") or d.get("has_wilor_2d")
                src[i] = C.SOURCE_IDS["observed" if obs else "interpolated"]
            if valid.any():
                tr.add_hand(side, go, hp, betas, tl, J, valid, src)
                tr.hands[f"hand_{side}"]["params_frame"] = np.array("camera_a_per_frame; left = mirrored right-hand params")
    tr.extra["T_world_cam"] = T_wc
    return tr
