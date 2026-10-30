"""Choose one reconstructed mesh per object from its SAM 3D candidates WITHOUT ground truth, and canonicalise it.

Candidates: recon_meshes/build/<obj>/view*/<obj>_seed*.obj (pick_views.py + spoon_mesh.py build; evaluation videos
only, metric scale from our stereo depth). Selection = consensus: the candidate whose shape is closest (symmetric
Chamfer after per-pair Sim(3) registration -- the kit's own CD-O registration code) to the other candidates
(median over them). A bad view / hallucinated part disagrees with the rest and loses. Metric scale = the median of
the candidates' stereo-derived extents (robust to one bad depth view).

Output: meshes_recon/<obj>/<obj>.obj -- canonical frame like the provided scans (origin = surface centroid, principal
axes; largest extent on z), + PROVENANCE.json. The public scans are NOT read here, except for an optional
development-only report (--dev_report) of the official CD-O of the chosen and of every candidate against the scan,
written separately and never used for the choice.
Usage: python select_mesh.py <obj> [--dev_report]   (kaggle venv: numpy, scipy, trimesh)
"""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path("/scratch/kghasemz/v2d")
KIT = ROOT / "submission_kit_new/unz/v2d_submission_kit"
_src = "\n".join(l for l in (KIT / "metric_code/track_3/CD-O.py").read_text().splitlines()
                 if not l.startswith(("import pandas", "from pandas")))
CDO = types.ModuleType("cdo"); CDO.pd = types.SimpleNamespace(DataFrame=object)
exec(compile(_src, "CD-O.py", "exec"), CDO.__dict__)


def chamfer_cm(a, b, seed=0):
    return float(CDO._object_chamfer_cm((np.asarray(a.vertices), np.asarray(a.faces)),
                                        (np.asarray(b.vertices), np.asarray(b.faces)), seed))


def canonical(m):
    m = m.copy()
    c = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
    cen = c.mean(0)
    w, U = np.linalg.eigh(np.cov((c - cen).T))  # ascending: smallest extent -> x, largest -> z
    if np.linalg.det(U) < 0:
        U[:, 0] *= -1
    T = np.eye(4); T[:3, :3] = U.T; T[:3, 3] = -U.T @ cen
    m.apply_transform(T)
    return m


def main():
    obj = sys.argv[1]
    cands = sorted((ROOT / "recon_meshes/build" / obj).glob(f"view*/{obj}_seed*.obj"))
    meshes = [trimesh.load(p, force="mesh") for p in cands]
    n = len(meshes)
    if n == 0:
        raise SystemExit(f"{obj}: no candidates")
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = chamfer_cm(meshes[i], meshes[j], 1000 * i + j)
    score = np.array([np.median(np.delete(D[i], i)) if n > 1 else 0.0 for i in range(n)])
    best = int(np.argmin(score))
    force = os.environ.get("SELECT_CAND")   # chosen by silhouette fit to OUR masks (pick_by_silhouette.py), not consensus
    if force:
        best = [str(p) for p in cands].index(force)
    ext = np.array([np.sort(m.extents)[::-1] for m in meshes])
    target = np.median(ext, axis=0)                        # robust metric size over all candidates
    chosen = canonical(meshes[best])
    s = float(np.median(target / np.sort(chosen.extents)[::-1]))
    chosen.apply_scale(s)
    out = ROOT / "meshes_recon" / obj; out.mkdir(parents=True, exist_ok=True)
    chosen.export(out / f"{obj}.obj")
    (out / f"{obj}.orig.obj").unlink(missing_ok=True)   # render_scale/stereo_scale --apply rescale from .orig: never a stale one
    prov = {"object": obj, "source": "reconstructed from the EVALUATION videos (no provided scan used)",
            "method": "pick_views.py (own SAM 3 masks, unoccluded free-standing views) -> SAM 3D Objects (spoon_mesh.py "
                      "build), metric scale from stereo depth; consensus choice = min median Sim(3)-registered Chamfer "
                      "to the other candidates (kit CD-O registration); size = median candidate extent",
            "candidates": [str(p) for p in cands], "consensus_score_cm": score.round(3).tolist(), "chosen": str(cands[best]),
            "scale_to_median_extent": s, "extent_cm": (np.sort(chosen.extents)[::-1] * 100).round(1).tolist(),
            "frame": "origin = surface centroid, principal axes (largest extent on z)"}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    print(f"{obj}: {n} candidates, consensus {score.round(2).tolist()} -> #{best} {cands[best].parent.name}/{cands[best].name}, "
          f"extent {prov['extent_cm']} cm", flush=True)
    if "--dev_report" in sys.argv:  # development only: never feeds back into the choice above
        scan = None
        d = ROOT / "data/track_3/public/mesh" / obj
        for p in (d / f"{obj}.glb", d / f"{obj}_visual.glb"):
            if p.exists():
                scan = trimesh.load(p, force="mesh")
        if scan is not None:
            rep = {"chosen_cdo_cm": chamfer_cm(chosen, scan, 7),
                   "candidates_cdo_cm": [round(chamfer_cm(m, scan, 7), 3) for m in meshes],
                   "scan_extent_cm": (np.sort(scan.extents)[::-1] * 100).round(1).tolist()}
            (ROOT / "recon_meshes" / "dev_report").mkdir(parents=True, exist_ok=True)
            (ROOT / "recon_meshes/dev_report" / f"{obj}.json").write_text(json.dumps(rep, indent=1))
            print(f"   DEV ONLY vs public scan: chosen CD-O {rep['chosen_cdo_cm']:.3f} cm; candidates {rep['candidates_cdo_cm']}; "
                  f"scan extent {rep['scan_extent_cm']}", flush=True)


if __name__ == "__main__":
    main()
