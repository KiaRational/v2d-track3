"""Lengthen a reconstructed mesh at its bottom so it reaches the table (water pitcher: the pose fits the image -- the
top / body -- but the mesh is short, so the resting pitcher floats 36-41 mm above the table in eval ep20, public ep21
and ep22). The object's 'up' axis in its own frame = the table normal mapped into the object frame over untouched
resting frames (median over runs); gap = median lowest-point height above the table there. Vertices are stretched along
that axis about the TOP (top fixed, bottom moved down by the gap; linear in height), so every existing pose stays valid
for the top/body and the base now touches the table. Backup <name>.prestretch.obj (never compounded).
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python extend_mesh.py <obj> <run>... [--poses tpfsu_recon]
       [--frames 0:6] (resting frames per run, start:end) [--dry]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh

from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--poses", default="tpfsu_recon")
    ap.add_argument("--frames", default="0:6")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    mesh_dir = ROOT / os.environ.get("V2D_MESH_DIR", "meshes_recon") / a.obj
    src = mesh_dir / f"{a.obj}.prestretch.obj"
    if not src.exists():
        shutil.copy(mesh_dir / f"{a.obj}.obj", src)
    mesh = trimesh.load(src, force="mesh", process=False)
    V = np.asarray(mesh.vertices)
    f0, f1 = (int(x) for x in a.frames.split(":"))
    ups, gaps = [], []
    for rn in a.runs:
        run = ROOT / "runs" / rn
        tp = json.loads((run / "table_plane.json").read_text())
        up, h = np.asarray(tp["up"], float), float(tp["height"])
        for t in range(f0, f1):
            T = load_T(run / "recon_hawor/slam_poses" / f"{t:06d}.json") @ load_T(run / a.poses / a.obj / "poses" / f"{t:06d}.json")
            ups.append(T[:3, :3].T @ up)                       # world up in the object frame
            gaps.append(float(((V @ T[:3, :3].T + T[:3, 3]) @ up).min() - h))
    ax = np.mean(ups, 0)
    ax /= np.linalg.norm(ax)
    gap = float(np.median(gaps))
    hv = V @ ax
    top, bot = hv.max(), hv.min()
    L = top - bot
    s = (L + gap) / L
    V2 = V + ((top - hv) * (s - 1))[:, None] * (-ax)           # height below the top scaled by s: bottom moves down by gap
    print(json.dumps({"obj": a.obj, "up_axis_obj": np.round(ax, 3).tolist(), "axis_spread_deg":
                      round(float(np.degrees(np.arccos(np.clip(np.min([u @ ax / np.linalg.norm(u) for u in ups]), -1, 1)))), 1),
                      "gaps_mm": [round(g * 1000, 1) for g in gaps[:: max(1, len(gaps) // 6)]], "gap_mm": round(gap * 1000, 1),
                      "height_cm": round(L * 100, 1), "new_height_cm": round((L + gap) * 100, 1), "stretch": round(s, 3)}))
    if a.dry:
        return
    out = mesh.copy()          # keeps faces + texture coordinates
    out.vertices = V2
    out.export(mesh_dir / f"{a.obj}.obj")
    pv = mesh_dir / "PROVENANCE.json"
    if pv.exists():
        p = json.loads(pv.read_text())
        p["bottom_stretch"] = {"gap_mm": round(gap * 1000, 1), "axis_obj": ax.tolist(), "runs": a.runs,
                               "reason": "resting object floated above the table: mesh too short at the bottom"}
        pv.write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
