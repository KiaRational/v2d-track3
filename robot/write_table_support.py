"""Our reconstructed round table -> the FlashCHORD support surface of a converted sequence (rg image).

FlashCHORD loads `<hmd>/ego_recon/reconstructed_stage/<seq>_support.usda` (Cylinder/Cube prims) as a static
collision body with the scene friction (contact_friction). Upstream reconstruct_support_surfaces.py drops every
disk within 5 cm of z = 0 as "the ground plane", and our table top is z = 0, so it always wrote an empty file:
the sim had only an infinite ground plane at table height, which blocks fingers from going under the rim
(e.g. a plank overhanging the table edge, ep23/24). This writes the table as one finite Cylinder ("table":
radius from table_plane.json, nominal top thickness, top face at the table height) so training can run with
collision.ground=false and robot-support collision on; upstream prims, if any, are kept.

Sim frame: the export's table frame (z = 0 on the table top) moved by the loader's pure translation
t = (-mean x, -mean y, -min z + ZOFF) of all bodies' frame-0 mesh vertices (no rotation: --no_ground_align).
t is recomputed exactly as the loader does, then cross-checked against the parquet object trajectory:
p_parquet(f) = p_bundle(f) + R_bundle(f) c_b + d, solved for d (and per-body origin offsets c_b); |d - t| and
the residual must be small, else the sequence is refused.
Usage: python write_table_support.py <clip_dir> <seq> [--zoff Z] [--robot sharpa_wave]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from robotic_grounding.retarget.support_recon import _load_parquet_data


def loader_translation(result_dir, npz, zoff):
    """ego_recon_loader._compute_alignment with --no_ground_align (R = I)."""
    if "object_names" in npz.files:
        bodies = [(result_dir / f"mesh_{i}.obj", float(npz[f"object_{i}_scale"]), f"object_{i}_to_world_transform")
                  for i in range(len(npz["object_names"]))]
    else:
        bodies = [(result_dir / "mesh.obj", float(npz["object_scale"]), "object_to_world_transform")]
    vw0 = []
    for mesh_path, scale, key in bodies:
        T0 = npz[key][0].astype(np.float64)
        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts = np.asarray(mesh.vertices, dtype=np.float64) * scale
        vw0.append((T0[:3, :3] @ verts.T).T + T0[:3, 3])
    vw0 = np.concatenate(vw0)
    return np.array([-vw0[:, 0].mean(), -vw0[:, 1].mean(), -vw0[:, 2].min() + zoff]), [b[2] for b in bodies]


def cross_check(npz, keys, data):
    """Solve p_parquet = p_bundle + R_bundle c_b + d over all frames/bodies -> d, residual (m), rot err (deg)."""
    P = np.asarray(data.object_body_position, np.float64)
    Q = np.asarray(data.object_body_wxyz, np.float64)
    P, Q = P.reshape(P.shape[0], -1, 3), Q.reshape(Q.shape[0], -1, 4)
    n, nb = min(P.shape[0], npz[keys[0]].shape[0]), len(keys)
    if P.shape[1] != nb:
        raise ValueError(f"parquet has {P.shape[1]} object bodies, bundle {nb}")
    rows, rhs, rot_err = [], [], []
    for b, k in enumerate(keys):
        T = npz[k][:n].astype(np.float64)
        Rq = Rotation.from_quat(Q[:n, b][:, [1, 2, 3, 0]]).as_matrix()
        rot_err.append(np.degrees(Rotation.from_matrix(np.einsum("nji,njk->nik", T[:, :3, :3], Rq)).magnitude()))
        for f in range(n):
            A = np.zeros((3, 3 + 3 * nb))
            A[:, :3] = np.eye(3)
            A[:, 3 + 3 * b:6 + 3 * b] = T[f, :3, :3]
            rows.append(A)
            rhs.append(P[f, b] - T[f, :3, 3])
    A, y = np.concatenate(rows), np.concatenate(rhs)
    # small ridge on the body-origin offsets: static bodies make c_b and d only jointly observable
    reg = np.zeros((3 * nb, 3 + 3 * nb))
    reg[:, 3:] = 1e-3 * np.eye(3 * nb)
    sol = np.linalg.lstsq(np.r_[A, reg], np.r_[y, np.zeros(3 * nb)], rcond=None)[0]
    res = np.linalg.norm((A @ sol - y).reshape(-1, 3), axis=1)
    return sol[:3], sol[3:].reshape(nb, 3), float(np.percentile(res, 95)), float(np.max(np.concatenate(rot_err)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("seq")
    ap.add_argument("--zoff", type=float, default=None, help="loader --ground_z_offset (default: manifest value)")
    ap.add_argument("--robot", default="sharpa_wave")
    ap.add_argument("--tol", type=float, default=0.003, help="max |d - t| and p95 residual (m)")
    ap.add_argument("--floor_depth", type=float, default=0.75,
                    help="floor (Cube) top this far below the table top, so an object knocked off the table lands "
                         "instead of falling forever (collision.ground=false); 0 = no floor")
    a = ap.parse_args()
    result_dir = Path(a.clip) / "outputs/result"
    man = json.loads((result_dir / "manifest.json").read_text())
    table = man.get("table")
    if not table:
        raise SystemExit(f"{a.seq}: manifest has no table (table_plane.json disk fit missing / not ok)")
    zoff = man.get("loader_ground_z_offset", 0.0) if a.zoff is None else a.zoff
    hmd = Path(os.environ["HUMAN_MOTION_DATA_DIR"]) / "ego_recon"
    with np.load(result_dir / "result.npz") as npz:
        t, keys = loader_translation(result_dir, npz, zoff)
        data = _load_parquet_data(hmd / "processed", a.seq, "mano_sharpa", a.robot)
        d, c, res95, rot = cross_check(npz, keys, data)
    print(f"{a.seq}: loader translation (recomputed) {np.round(t, 4)}, from parquet {np.round(d, 4)} "
          f"(|diff| {np.linalg.norm(d - t) * 1000:.2f} mm, p95 residual {res95 * 1000:.2f} mm, max rot {rot:.3f} deg, "
          f"body origin offsets {np.round(np.linalg.norm(c, axis=1) * 1000, 1)} mm)", flush=True)
    if np.linalg.norm(d - t) > a.tol or res95 > a.tol or rot > 0.5:
        raise SystemExit(f"{a.seq}: sim frame cross-check failed -- not writing a table")
    ctr = np.asarray(table["center"], float) + d          # parquet-measured translation (== t within tol)
    r, th = float(table["radius"]), float(table["thickness"])

    path = hmd / "reconstructed_stage" / f"{a.seq}_support.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    keep, old = [], None
    if path.exists():
        old = Usd.Stage.Open(str(path))  # keep a reference: a temporary stage expires mid-traversal
        for p in old.Traverse():
            if p.GetTypeName() in ("Cylinder", "Cube") and p.GetName() not in ("table", "floor"):
                keep.append(p.GetPath())
        if keep:
            print(f"keeping {len(keep)} upstream support prims: {[str(p) for p in keep]}")
    tmp = path.with_suffix(".tmp.usda")
    stage = Usd.Stage.CreateNew(str(tmp))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/support_surfaces")
    stage.SetDefaultPrim(root.GetPrim())
    cyl = UsdGeom.Cylinder.Define(stage, "/support_surfaces/table")
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateHeightAttr(th)
    cyl.CreateRadiusAttr(r)
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set((float(ctr[0]), float(ctr[1]), float(ctr[2] - th / 2)))
    if a.floor_depth > 0:  # far below the rim: hands still reach under the table; 4 m wide
        L = 4.0
        fl = UsdGeom.Cube.Define(stage, "/support_surfaces/floor")
        fl.CreateSizeAttr(L)
        UsdPhysics.CollisionAPI.Apply(fl.GetPrim())
        UsdGeom.Xformable(fl.GetPrim()).AddTranslateOp().Set((float(ctr[0]), float(ctr[1]), float(ctr[2] - a.floor_depth - L / 2)))
    for p in keep:  # copy the upstream prim spec as-is
        Sdf.CopySpec(old.GetRootLayer(), p, stage.GetRootLayer(), p)
    stage.GetRootLayer().Save()
    tmp.replace(path)
    info = {"seq": a.seq, "center_sim": ctr.tolist(), "radius": r, "thickness": th, "top_z_sim": float(ctr[2]),
            "loader_translation": t.tolist(), "parquet_translation": d.tolist(), "p95_residual_m": res95,
            "kept_upstream_prims": [str(p) for p in keep], "floor_top_z_sim": float(ctr[2] - a.floor_depth) if a.floor_depth > 0 else None, "disk_fit": table.get("disk_fit")}
    (path.parent / f"{a.seq}_table.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {path}: table r={r:.4f} m, thickness {th} m, top z {ctr[2]:.4f}, centre xy {np.round(ctr[:2], 4)}")


if __name__ == "__main__":
    main()
