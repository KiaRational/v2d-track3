"""Sim-validity pass on a training reference: no object may start (or rest) inside another object.

Found on 6/40 episodes (sweeper in dish rack 12 mm, pot lid in pot 8 mm, pot in rack 9 mm at frame 0): the
reconstruction tolerates small object-object overlaps, nothing downstream removed them (depenetrate.py handles
object-vs-table only), and the simulator's contact solver then locks the pair together -- the policy cannot move
the object at all (eval ep30: the sweeper never moves).

Uses the SIMULATOR's own collision geometry: each object's URDF collision mesh convex-decomposed exactly as
FlashCHORD does (flash_chord.assets.mesh.convex_decompose, CoACD, its default threshold), so a corrected reference
is valid in the very geometry the policy is trained and evaluated with. Per frame and pair, the SMALLER object (by
convex-part volume) is pushed out along the mean exit normal of the parts its surface is inside, by the depth plus
a margin, iterated until clear; it is never pushed below the table top. Corrections are box-filtered in time (no
jumps) and the pass is repeated on the smoothed result so every frame ends clear. Only object positions change
(orientation untouched); frames that are already valid are not modified beyond the smoothing ramp.

Usage (flash_chord image): python sim_validity.py <hmd_root> <seq>[,<seq>...]   (rewrites the seq's parquet)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull as QH
from scipy.spatial.transform import Rotation as Rot

from flash_chord.assets.mesh import convex_decompose

TOL, MARGIN, TABLE_R = 0.0015, 0.0015, 0.25
MAX_MOVE = 0.012  # cap: a real resting-pose overlap is a few mm; larger moves would leave the truth (pot lids pushed 25 mm)
CACHE = "/scratch/kghasemz/v2d/cache/coacd_fc"


def coll_meshes(urdf):
    """Every <collision> mesh listed in the URDF (one, or the pre-split pieces of fine_collision.py)."""
    txt = Path(urdf).read_text()
    ps = [Path(m) for m in re.findall(r"<collision>.*?filename=\"([^\"]+)\"", txt, re.S)]
    return [p if p.is_absolute() else (Path(urdf).parent / p).resolve() for p in ps]


def parts(meshes):
    """The simulator's pieces: each listed mesh convex-decomposed with FlashCHORD's defaults (a pre-split convex
    piece stays one piece)."""
    hs = [h for mesh in meshes for h in convex_decompose(np.asarray(mesh.vertices), np.asarray(mesh.faces), cache_dir=CACHE)]
    out = []
    for h in hs:
        if len(h.vertices) >= 4:
            q = QH(h.vertices)
            out.append((q.equations, q.volume))
    return out


def push_dir(P, prts):
    """For points P (in the other object's frame): max depth inside the union of parts and the mean exit normal."""
    depth, normals = 0.0, []
    for E, _ in prts:
        s = P @ E[:, :3].T + E[:, 3]
        inside = (s < 0).all(1)
        if inside.any():
            si = s[inside]
            k = np.argmax(si, axis=1)               # exit face = the least-penetrated plane
            d = -si[np.arange(len(si)), k]
            depth = max(depth, float(d.max()))
            normals.append((E[k, :3] * d[:, None]).sum(0))
    if not normals:
        return 0.0, None
    n = np.sum(normals, 0)
    return depth, n / (np.linalg.norm(n) + 1e-12)


def _pen_batch(Pm, Rm, tm, Ro, to, prts):
    """Max penetration of mover points Pm (mover frame) at pose (Rm, tm) into the other object's parts."""
    Po = ((Pm @ Rm.T + tm) - to) @ Ro
    return push_dir(Po, prts)[0]


def search_fix(Pm, Rm, tm, Ro, to, prts, max_t=MAX_MOVE, step=0.003, max_deg=5.0):
    """Smallest rigid move (translation, then translation + small rotation) that clears the overlap.
    Used when normal pushing does not converge (object wedged between walls of a slot / opening)."""
    sub = Pm[:: max(1, len(Pm) // 500)]
    g = np.arange(-max_t, max_t + 1e-9, step)
    cand = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    cand = cand[np.argsort(np.linalg.norm(cand, axis=1))]
    rots = [np.eye(3)] + [Rot.from_rotvec(np.deg2rad(a) * np.asarray(ax)).as_matrix()
                          for a in (max_deg / 2, max_deg) for s in (1, -1) for ax in ((s, 0, 0), (0, s, 0), (0, 0, s))]
    for dR in rots:                       # prefer no rotation, then the smallest rotations
        for d in cand:
            if _pen_batch(sub, Rm @ dR, tm + d, Ro, to, prts) < TOL and _pen_batch(Pm, Rm @ dR, tm + d, Ro, to, prts) < TOL:
                return d, dR
    return None, None


def separate(pos, R, S, P, mover_of, active, max_iter=12):
    """One pass over the ACTIVE frames (each mover's initial rest); returns per-object correction (F, K, 3)."""
    F, K = pos.shape[:2]
    corr = np.zeros_like(pos)
    for f in range(F):
        for (a, b), m in mover_of.items():
            if not active[f, m]:
                continue
            o = b if m == a else a
            for _ in range(max_iter):
                Pw = S[m] @ R[f, m].T + pos[f, m] + corr[f, m]
                Po = (Pw - pos[f, o] - corr[f, o]) @ R[f, o]
                d, n = push_dir(Po, P[o])
                if d < TOL:
                    break
                step = (R[f, o] @ n) * (d + MARGIN)
                Pn = Pw + step
                low = Pn[np.linalg.norm(Pn[:, :2], axis=1) < TABLE_R, 2]
                if low.size and low.min() < 0:      # never into the table: keep the push horizontal-up
                    step[2] = max(step[2], -low.min() + step[2])
                corr[f, m] += step
                nrm = np.linalg.norm(corr[f, m])
                if nrm > MAX_MOVE:                  # capped: report rather than move far from the reconstruction
                    corr[f, m] *= MAX_MOVE / nrm
                    break
    return corr


def run(hmd, seq):
    pqf = next((Path(hmd) / f"ego_recon/processed/sequence_id={seq}").rglob("*.parquet"))
    t = pq.read_table(pqf)
    names = t.column("object_body_names")[0].as_py()
    if len(names) < 2:
        print(f"{seq}: single object, nothing to do"); return
    urdfs = t.column("object_urdf_paths")[0].as_py()
    pos = np.asarray(t.column("object_body_position")[0].as_py(), float)
    q = np.asarray(t.column("object_body_wxyz")[0].as_py(), float)
    R = np.stack([Rot.from_quat(q[:, k][:, [1, 2, 3, 0]]).as_matrix() for k in range(len(names))], 1)
    pieces = [[trimesh.load(p, force="mesh") for p in coll_meshes(u)] for u in urdfs]
    meshes = [trimesh.util.concatenate(ps) if len(ps) > 1 else ps[0] for ps in pieces]
    P = [parts(ps) for ps in pieces]
    S = [trimesh.sample.sample_surface(m, 2000, seed=0)[0] for m in meshes]
    vol = [sum(v for _, v in p) for p in P]
    mover_of = {(a, b): (a if vol[a] <= vol[b] else b) for a in range(len(names)) for b in range(a + 1, len(names))}
    # scope: an object must START valid -- the policy cannot undo a lock that exists before the hand arrives.
    # Only each object's initial rest (until it first moves > 5 mm / 3 deg) is corrected; overlaps during
    # manipulation (a cup being nested, a spoon put into the rack) are left to the simulator's contact solver.
    F = len(pos)
    rest_end = []
    for k in range(len(names)):
        dp = np.linalg.norm(pos[:, k] - pos[0, k], axis=1)
        dr = np.degrees(Rot.from_matrix(np.einsum("ji,fjk->fik", R[0, k], R[:, k])).magnitude())
        mv = np.where((dp > 0.005) | (dr > 3.0))[0]
        rest_end.append(int(mv[0]) if len(mv) else F)
    active = np.zeros((F, len(names)), bool)
    for k, e in enumerate(rest_end):
        active[:e, k] = True
    total = np.zeros_like(pos)
    for rnd in range(3):
        c = separate(pos + total, R, S, P, mover_of, active)
        if not np.abs(c).any():
            break
        total += uniform_filter1d(c, size=5, axis=0, mode="nearest") if rnd < 2 else c
    total += separate(pos + total, R, S, P, mover_of, active)  # final unsmoothed pass: every frame clear
    nt = np.linalg.norm(total, axis=-1, keepdims=True)  # the cap holds for the accumulated correction too
    total *= np.minimum(1.0, MAX_MOVE / np.maximum(nt, 1e-12))
    # fallback for what pushing could not clear (wedged in a slot / opening): one rigid move per contiguous run
    newR = R.copy()
    F = len(pos)
    fixes = []
    for (a, b), m in mover_of.items():
        o = b if m == a else a
        bad = np.array([active[f, m] and _pen_batch(S[m], newR[f, m], pos[f, m] + total[f, m], R[f, o], pos[f, o] + total[f, o], P[o]) >= TOL
                        for f in range(F)])
        f = 0
        while f < F:
            if not bad[f]:
                f += 1; continue
            e = f
            while e + 1 < F and bad[e + 1]:
                e += 1
            d, dR = search_fix(S[m], newR[f, m], pos[f, m] + total[f, m], R[f, o], pos[f, o] + total[f, o], P[o])
            if d is not None:
                ramp = 5
                for g in range(max(0, f - ramp), min(F, e + ramp + 1)):
                    w = 1.0 if f <= g <= e else 1 - min(abs(g - f), abs(g - e)) / (ramp + 1)
                    total[g, m] += w * d
                    newR[g, m] = newR[g, m] @ Rot.from_rotvec(w * Rot.from_matrix(dR).as_rotvec()).as_matrix()
                fixes.append({"mover": names[m], "frames": [int(f), int(e)], "move_mm": round(float(np.linalg.norm(d)) * 1000, 1),
                              "rot_deg": round(float(np.degrees(Rot.from_matrix(dR).magnitude())), 1)})
            else:
                fixes.append({"mover": names[m], "frames": [int(f), int(e)], "unresolved": True})
            f = e + 1
    # outside each object's initial rest: the correction ramps out over 5 frames (no jump), then is zero
    for k, e in enumerate(rest_end):
        if 0 < e < F:
            c_end = total[e - 1, k].copy()
            rv = Rot.from_matrix(R[e - 1, k].T @ newR[e - 1, k]).as_rotvec()
            for g in range(e, F):
                w = max(0.0, 1 - (g - e + 1) / 6)
                total[g, k] = w * c_end
                newR[g, k] = R[g, k] @ Rot.from_rotvec(w * rv).as_matrix()
    nt = np.linalg.norm(total, axis=-1, keepdims=True)  # strict cap, after every step
    total *= np.minimum(1.0, MAX_MOVE / np.maximum(nt, 1e-12))
    if fixes:
        print(f"{seq}: search fixes {fixes}", flush=True)
    mag = np.linalg.norm(total, axis=-1) + (np.abs(newR - R).sum((-1, -2)) > 1e-9) * 1e-3
    rep = {names[k]: {"frames_moved": int((mag[:, k] > 1e-4).sum()), "max_mm": round(float(mag[:, k].max()) * 1000, 1)}
           for k in range(len(names))}
    print(f"{seq}: {rep}", flush=True)
    if not (mag > 1e-4).any():
        return
    new = pos + total
    root = np.asarray(t.column("object_root_position")[0].as_py(), float)
    root_is_body0 = np.allclose(root, pos[:, 0], atol=1e-6)
    cols = {}
    cols["object_body_position"] = pa.array([new.tolist()], type=t.schema.field("object_body_position").type)
    if (np.abs(newR - R) > 1e-9).any():
        xyzw = np.stack([Rot.from_matrix(newR[:, k]).as_quat() for k in range(newR.shape[1])], 1)
        cols["object_body_wxyz"] = pa.array([xyzw[..., [3, 0, 1, 2]].tolist()], type=t.schema.field("object_body_wxyz").type)
        if root_is_body0:
            cols["object_root_axis_angle"] = pa.array([Rot.from_matrix(newR[:, 0]).as_rotvec().tolist()],
                                                      type=t.schema.field("object_root_axis_angle").type)
    if root_is_body0:
        cols["object_root_position"] = pa.array([new[:, 0].tolist()], type=t.schema.field("object_root_position").type)
    arrays = [cols.get(n, t.column(n)) for n in t.column_names]
    md = dict(t.schema.metadata or {}); md[b"v2d_sim_validity"] = str(rep).encode()
    pq.write_table(pa.Table.from_arrays(arrays, schema=t.schema.with_metadata(md)), pqf)


if __name__ == "__main__":
    for s in sys.argv[2].split(","):
        run(sys.argv[1], s)
