"""Convex parts of the provided Track 3 collision meshes, as FlashCHORD will decompose them (CoACD, its default
threshold 0.05) -> <out>/<object>.npz (n, v0..v{n-1}: part vertices in the object frame). The export reads them
to check object-object overlap with the simulator's geometry (e.g. the slotted wooden pieces, whose slots the
decomposition of the scanned visual mesh fills in). Run in the flash_chord image (has coacd).
Usage: python collision_parts.py <mesh_root> <out_dir> [object ...]"""
import sys
from pathlib import Path

import coacd
import numpy as np
import trimesh

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
names = sys.argv[3:] or sorted(p.name for p in root.iterdir() if p.is_dir())
coacd.set_log_level("error")
for o in names:
    src = root / o / f"{o}_collision.glb"
    if not src.exists():
        print(f"{o}: no provided collision mesh")
        continue
    m = trimesh.load(src, force="mesh")
    parts = [v for v, _ in coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=0.05)]
    np.savez(out / f"{o}.npz", n=len(parts), **{f"v{k}": v for k, v in enumerate(parts)})
    print(f"{o}: {len(parts)} parts", flush=True)
