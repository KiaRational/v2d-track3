"""Fine collision geometry for every object (flash_chord image): pre-split convex pieces in the URDF.

Measured on 6/40 episodes: FlashCHORD convex-decomposes each object's collision mesh with CoACD at its default
threshold (0.05), which partly fills container openings (pot rim, dish-rack slots): a lid resting on its pot or a
sweeper standing in a rack slot then starts 3-6 mm "inside" the other object purely from the approximation, and the
simulator locks the pair. The pinned official evaluator always uses that default, but the decomposition runs on
whatever collision meshes the URDF lists: listing the object as many already-convex pieces (CoACD at a finer
threshold, done here once) makes the simulator keep those pieces (a convex piece decomposes into itself), in
training and in the unchanged official evaluation alike.

Source mesh per object: the dataset's provided *_collision mesh where one exists (data/track_3/public/mesh), else
the URDF's collision mesh. Writes <dst>/ego_recon/processed/collision_parts/<obj>/part_NNN.stl and rewrites
<dst>/.../<obj>_rigid.urdf (copied from <src>, inertial kept) with one <collision> element per piece.
Usage: python fine_collision.py <src_hmd_root> <dst_hmd_root> [threshold=0.02]
"""
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

from flash_chord.assets.mesh import convex_decompose

PROVIDED = Path("/scratch/kghasemz/v2d/data/track_3/public/mesh")
CACHE = "/scratch/kghasemz/v2d/cache/coacd_fc"
import os
SHELL = float(os.environ.get("SHELL_MM", "0")) / 1000.0  # >0: walls-only collision (shell thickness ~ this)
# Per-object collision (env SHELL_MAP, "name:mm,..."): walls only for CONTAINERS that must stay open, with their own
# wall thickness (thin 4 mm walls let fingers tunnel into cups and made every object rock on the table); every other
# object is decomposed SOLID. FINE_TH_MAP "name:th,...": finer decomposition (keeps the boards' assembly slots open).
SHELL_MAP = {k: float(v) / 1000.0 for k, v in (x.split(":") for x in os.environ.get("SHELL_MAP", "").split(",") if x)}
TH_MAP = {k: float(v) for k, v in (x.split(":") for x in os.environ.get("FINE_TH_MAP", "").split(",") if x)}


def shell_of(mesh, t):
    """Thin watertight shell (~t thick) around the surface: surface voxels at pitch t, marching cubes, then the
    shell's own surface. Unlike the mesh's solid outline, its convex pieces never cover an opening."""
    vox = mesh.voxelized(pitch=t)          # surface voxels only (not filled)
    sh = vox.marching_cubes                # in voxel index space
    sh.apply_transform(vox.transform)
    if len(sh.faces):   # keep the largest face-connected component (trimesh's split() needs networkx: not in the image)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        adj = sh.face_adjacency
        nf = len(sh.faces)
        g = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(nf, nf))
        _, lab = connected_components(g, directed=False)
        keep = lab == np.bincount(lab).argmax()
        sh.update_faces(keep)
        sh.remove_unreferenced_vertices()
    return sh


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    sp, dp = src / "ego_recon/processed", dst / "ego_recon/processed"
    dp.mkdir(parents=True, exist_ok=True)
    for e in (src / "ego_recon").iterdir():  # mirror the tree (symlinks) except processed/, built below
        if e.name != "processed" and not (dst / "ego_recon" / e.name).exists():
            (dst / "ego_recon" / e.name).symlink_to(e.resolve())
    for e in sp.iterdir():
        t = dp / e.name
        if e.name.endswith("_rigid.urdf") or e.name.startswith("sequence_id=") or t.exists() or t.is_symlink():
            continue
        t.symlink_to(e.resolve())
    for u in sorted(sp.glob("*_rigid.urdf")):
        name = u.name[: -len("_rigid.urdf")]
        txt = u.read_text()
        m = re.search(r"<collision>.*?filename=\"([^\"]+)\"", txt, re.S)
        mp = Path(m.group(1)); mp = mp if mp.is_absolute() else (u.parent / mp).resolve()
        prov = PROVIDED / name / f"{name}_collision.glb"
        if os.environ.get("V2D_PROVIDED_COLLISION", "all") == "none":   # compliant tree: never the provided mesh
            prov = Path("/nonexistent")
        mesh = trimesh.load(prov if prov.exists() else mp, force="mesh")
        shell = SHELL_MAP.get(name, 0.0) if SHELL_MAP else SHELL
        if shell > 0:  # walls only: a solid shell around the surface, so openings (pot mouth, rack slots) stay empty
            mesh = shell_of(mesh, shell)
        hulls = convex_decompose(np.asarray(mesh.vertices), np.asarray(mesh.faces), threshold=TH_MAP.get(name, th), cache_dir=CACHE)
        print(f"{name}: {'walls %.0f mm' % (shell * 1000) if shell > 0 else 'solid'}, threshold {TH_MAP.get(name, th)}", flush=True)
        od = dp / "collision_parts" / name
        shutil.rmtree(od, ignore_errors=True); od.mkdir(parents=True)
        cols = []
        for i, h in enumerate(hulls):
            pm = trimesh.Trimesh(h.vertices, h.faces).convex_hull
            pm.export(od / f"part_{i:03d}.stl")
            cols.append(f'      <collision>\n         <origin rpy="0 0 0" xyz="0 0 0"/>\n         <geometry>\n'
                        f'            <mesh filename="./collision_parts/{name}/part_{i:03d}.stl" scale="1 1 1"/>\n'
                        f'         </geometry>\n      </collision>')
        new = re.sub(r"\s*<collision>.*?</collision>", "", txt, flags=re.S)
        new = new.replace("</link>", "\n".join(cols) + "\n   </link>", 1)
        new = new.replace("<robot ", f"<!-- fine collision: {len(hulls)} convex pieces (CoACD {th}, source "
                                     f"{'provided collision mesh' if prov.exists() else 'URDF collision mesh'}) -->\n<robot ", 1)
        (dp / u.name).write_text(new)
        print(f"{name:18s} {len(hulls):4d} pieces  source {'provided' if prov.exists() else 'urdf'}", flush=True)


if __name__ == "__main__":
    main()
