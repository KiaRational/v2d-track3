"""Convex parts of the provided Track 3 collision meshes, as FlashCHORD will decompose them (CoACD, its default
threshold 0.05) -> <out>/<object>.npz (n, v0..v{n-1}: part vertices in the object frame). The export reads them
to check object-object overlap with the simulator's geometry (e.g. the slotted wooden pieces, whose slots the
decomposition of the scanned visual mesh fills in). Run in the flash_chord image (has coacd).
Usage: python collision_parts.py <mesh_root> <out_dir> [--visual] [object ...]
  --visual: decompose <obj>.obj (OUR reconstructed meshes, compliant tree) instead of the provided <obj>_collision.glb."""
import sys
from pathlib import Path

import coacd
import numpy as np
import trimesh

visual = "--visual" in sys.argv
argv = [x for x in sys.argv if x != "--visual"]
root, out = Path(argv[1]), Path(argv[2])
out.mkdir(parents=True, exist_ok=True)
names = argv[3:] or sorted(p.name for p in root.iterdir() if p.is_dir())
coacd.set_log_level("error")
for o in names:
    src = root / o / (f"{o}.obj" if visual else f"{o}_collision.glb")
    if not src.exists():
        print(f"{o}: no mesh {src.name}")
        continue
    m = trimesh.load(src, force="mesh")
    parts = [v for v, _ in coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=0.05)]
    np.savez(out / f"{o}.npz", n=len(parts), **{f"v{k}": v for k, v in enumerate(parts)})
    print(f"{o}: {len(parts)} parts", flush=True)
