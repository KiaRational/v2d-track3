"""Use the provided Track 3 collision mesh as the simulator collision geometry of selected objects (rg image).

upstream generate_rigid_urdfs.py writes processed/<obj>_rigid.urdf with the scanned visual mesh as the collision
mesh too. FlashCHORD convex-decomposes it (CoACD) and, for the slotted wooden pieces, the decomposition of the
scan fills the 1.8 cm slots (the scan's slot is narrow and ragged), so the two pieces cannot be slotted into
their X and the simulator pushes them apart. The provided *_collision.glb keeps the slots through the same
decomposition, and is in the same object frame as the visual mesh (checked: < 1 mm). The URDF is per object (not
per sequence) and is rewritten on every convert, so this runs after it in run_convert.sbatch.
Usage: python write_collision_mesh.py <clip_dir> [--objects all (default: every object with a provided mesh) | a,b] [--mesh_root R]
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--objects", default="all")  # every object the dataset ships a *_collision mesh for
    ap.add_argument("--mesh_root", default=os.path.join(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"),
                                                         "data/track_3/public/mesh"))
    ap.add_argument("--max_dist_mm", type=float, default=3.0, help="refuse if the meshes are not in one frame")
    a = ap.parse_args()
    result = Path(a.clip) / "outputs/result"
    man = json.loads((result / "manifest.json").read_text())
    with np.load(result / "result.npz") as z:
        names = [str(x) for x in z["object_names"]] if "object_names" in z.files else [man["sources"]["object"]]
    want = set(names) if a.objects == "all" else set(a.objects.split(","))
    proc = Path(os.environ["HUMAN_MOTION_DATA_DIR"]) / "ego_recon" / "processed"
    for o in names:
        src = Path(a.mesh_root) / o / f"{o}_collision.glb"
        urdf = proc / f"{o}_rigid.urdf"
        if o not in want or not src.exists() or not urdf.exists():
            continue
        col = trimesh.load(src, force="mesh")
        vis = trimesh.load(proc / f"{o}_visual.stl", force="mesh")
        d = cKDTree(vis.vertices).query(col.sample(5000))[0]
        if np.median(d) * 1000 > a.max_dist_mm:
            raise SystemExit(f"{o}: provided collision mesh is {np.median(d) * 1000:.1f} mm from the visual mesh -- "
                             "not in the same frame, refusing")
        out = proc / f"{o}_collision.stl"
        col.export(out)
        text = urdf.read_text()
        new = re.sub(r"(<collision>.*?<mesh filename=\")[^\"]+(\")", rf"\g<1>./{out.name}\g<2>", text, flags=re.S)
        if new == text and f"./{out.name}" not in text:
            raise SystemExit(f"{o}: no collision mesh element in {urdf}")
        urdf.write_text(new)
        print(f"{o}: collision mesh -> {out.name} ({len(col.faces)} faces, median {np.median(d) * 1000:.2f} mm "
              "from the visual mesh)")


if __name__ == "__main__":
    main()
