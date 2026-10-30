"""Give cups a solid base in the simulator: one extra convex collision piece = hull of the cup mesh's bottom BASE_M
along its own up axis. Walls-only shells left an upright cup standing on a few thin pieces (small, uneven contact
patch): in the passive test cups toppled at random (non-repeatable GPU physics), e.g. eval ep10 cups 85 deg in 3 s.
The cup stays open (spoon stirring, stacking). Up axis = the mesh's symmetry axis, signed by a resting pose (world up
mapped into the object frame). Writes <tree>/ego_recon/processed/<cup>_rigid.urdf as a real file (+ base piece in
collision_parts_base/<cup>/base.stl) and points the given sequences' parquets (and .presettle backups) at it.
Usage: V2D_MESH_DIR=meshes_recon python cup_base.py <tree> <run_for_up> <seq>..."""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T  # noqa: E402
from symmetry import mesh_symmetry  # noqa: E402

ROOT = Path("/scratch/kghasemz/v2d")
BASE_M = 0.008
CUPS = ("blue_cup", "beige_cup")


def main():
    tree, run_up, seqs = Path(sys.argv[1]), ROOT / "runs" / sys.argv[2], sys.argv[3:]
    proc = tree / "ego_recon/processed"
    mesh_dir = ROOT / os.environ.get("V2D_MESH_DIR", "meshes_recon")
    tp = json.loads((run_up / "table_plane.json").read_text())
    new_urdf = {}
    for cup in CUPS:
        src = proc / f"{cup}_rigid.urdf"
        txt = Path(os.path.realpath(src)).read_text()
        m = trimesh.load(mesh_dir / cup / f"{cup}.obj", force="mesh")
        V = np.asarray(m.vertices)
        ax = np.asarray(mesh_symmetry(mesh_dir / cup / f"{cup}.obj")["cont_axis"], float)
        T = load_T(run_up / "recon_hawor/slam_poses/000000.json") @ load_T(run_up / "tpfsut_recon" / cup / "poses/000000.json")
        if (T[:3, :3] @ ax) @ np.asarray(tp["up"]) < 0:
            ax = -ax                                   # points up when the cup stands
        h = V @ ax
        base = trimesh.convex.convex_hull(V[h < h.min() + BASE_M])
        od = proc / "collision_parts_base" / cup
        od.mkdir(parents=True, exist_ok=True)
        base.export(od / "base.stl")
        block = ('      <collision>\n         <origin rpy="0 0 0" xyz="0 0 0"/>\n         <geometry>\n'
                 f'            <mesh filename="{od / "base.stl"}" scale="1 1 1"/>\n         </geometry>\n      </collision>\n')
        # absolute filenames for the existing pieces (the copy lives in a different directory than the original)
        base_dir = Path(os.path.realpath(src)).parent
        txt = re.sub(r'filename="\./([^"]+)"', lambda mm: f'filename="{base_dir / mm.group(1)}"', txt)
        txt = txt.replace("</collision>", "</collision>" + "\n" + block, 1) if "</collision>" in txt else txt
        if src.is_symlink():
            src.unlink()
        src.write_text(txt)
        new_urdf[cup] = str(src)
        print(cup, "base piece: %d verts, height %.1f mm, footprint %.1f cm2" % (
            len(base.vertices), np.ptp(base.vertices @ ax) * 1000, base.area / 2 * 1e4), flush=True)
    for seq in seqs:
        for f in glob.glob(f"{proc}/sequence_id={seq}/robot_name=sharpa_wave/*.parquet*"):
            t = pq.read_table(f)
            names = t.column("object_body_names")[0].as_py()
            paths = t.column("object_urdf_paths")[0].as_py()
            paths2 = [new_urdf.get(nm, p) for nm, p in zip(names, paths)]
            if paths2 != paths:
                cols = {c: t.column(c) for c in t.column_names}
                cols["object_urdf_paths"] = pa.array([paths2], type=t.schema.field("object_urdf_paths").type)
                pq.write_table(pa.table(cols, schema=t.schema), f)
                print(seq, os.path.basename(f), "urdf ->", [p for p in paths2 if p in new_urdf.values()], flush=True)


if __name__ == "__main__":
    main()
