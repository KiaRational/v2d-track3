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
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

from flash_chord.assets.mesh import convex_decompose

PROVIDED = Path("/scratch/kghasemz/v2d/data/track_3/public/mesh")
CACHE = "/scratch/kghasemz/v2d/cache/coacd_fc"


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
        mesh = trimesh.load(prov if prov.exists() else mp, force="mesh")
        hulls = convex_decompose(np.asarray(mesh.vertices), np.asarray(mesh.faces), threshold=th, cache_dir=CACHE)
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
