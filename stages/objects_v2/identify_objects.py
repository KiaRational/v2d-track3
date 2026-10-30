"""Which of the dataset's objects appear in an episode, from the video alone (sam3 image).

The evaluation split ships only videos -- no object list -- and every later stage needs
one. SAM 3 is prompted with every object in the vocabulary on frames sampled across the
episode; a detection only counts when its metric 3D extent, measured from the stereo depth
under its mask, fits that object's scanned mesh (geom.geometry_check / fit_score), so a
cup cannot vote for "water pitcher" and the table cannot vote for "wooden board".

Per object: support = number of sampled frames with a fitting detection, weighted by
score x fit. Decisions use co-occurrence structure seen in the public split:
  * look-alike pairs (wooden_piece_1/2) share one prompt; the pair is present when two
    distinct fitting instances are seen in the same frame, and both names are reported;
  * white_pot_lid is only reported together with white_pot (it is otherwise easily
    confused with any white disc);
  * episodes track 1-2 objects (occasionally more): the top objects by support are kept,
    a second one only when its support is a clear fraction of the first's.

Writes <run>/objects_id.json {"objects": [...], "support": {...}, "frames": [...]}.
Usage (inside the sam3 image): python identify_objects.py <run> [...] [--frames 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import decode_depth, geometry_check, mask_iou  # noqa: E402
from seg_sam3 import Sam3Segmenter  # noqa: E402

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
PROMPTS = {
    "white_pot": ["white saucepan", "white pot"], "white_pot_lid": ["white pot lid"],
    "mini_sweeper": ["hand brush"], "mini_dust_pan": ["yellow dustpan"],
    "wooden_board": ["wooden board"],  # wooden_piece_1 / wooden_piece_2
    "plastic_dish_rack": ["dish rack"], "blue_cup": ["blue cup"], "beige_cup": ["beige cup"],
    "water_pitcher": ["water pitcher", "plastic jug", "jug", "pitcher", "watering can", "white watering can"],
}
PAIR = {"wooden_board": ["wooden_piece_1", "wooden_piece_2"]}


def obj_extents(path):
    """Axis-aligned extents of an OBJ's vertices (no trimesh in the sam3 image)."""
    V = [line.split()[1:4] for line in open(path) if line.startswith("v ")]
    V = np.asarray(V, float)
    return V.max(0) - V.min(0)


CONFUSABLE = [("white_pot", "water_pitcher")]  # answer to each other's prompts; never seen together


def decide(support, pair_frames, a):
    """Object list from per-label support. Rules, all from the public split's structure:
      * a confusable pair keeps only its better-supported member (flagged when close);
      * the boards count only as a pair actually seen twice together;
      * the lid only with the pot;
      * 1-2 object groups; a third needs >= --third_ratio of the top support."""
    sup = dict(support)
    notes = []
    lid = sup.get("white_pot_lid", 0.0)
    # The lid is part of the pot's evidence (a pitcher has no lid): it backs the pot in the
    # pot/pitcher decision and ranks with it as one group; it is listed as its own object only
    # when it is itself strong -- handled, not merely sitting there.
    grp = dict(sup)
    grp["white_pot"] = sup.get("white_pot", 0.0) + lid
    grp.pop("white_pot_lid", None)
    for x, y in CONFUSABLE:
        if grp.get(x, 0) >= a.min_support and grp.get(y, 0) >= a.min_support:
            hi, lo = (x, y) if grp[x] >= grp[y] else (y, x)
            if grp[lo] > (1 - a.close_margin) * grp[hi]:
                notes.append({"ambiguous": [hi, lo], "support": [round(grp[hi], 2), round(grp[lo], 2)]})
            grp[lo] = 0.0
    if pair_frames < 2:
        grp["wooden_board"] = 0.0
    ranked = sorted(((v, k) for k, v in grp.items() if v >= a.min_support), reverse=True)
    groups = []
    for v, k in ranked:
        if not groups or (len(groups) == 1 and v >= a.second_ratio * ranked[0][0]) or \
                (len(groups) >= 2 and v >= a.third_ratio * ranked[0][0]):
            groups.append(k)
    objects = []
    for k in groups:
        objects += PAIR.get(k, [k])
    if "white_pot" in objects and ranked and lid >= a.third_ratio * ranked[0][0]:
        objects.append("white_pot_lid")
    return sorted(objects), notes


def frames_from_video(path, idx):
    cap, out = cv2.VideoCapture(str(path)), {}
    for t in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, f = cap.read()
        if ok:
            out[int(t)] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--min_support", type=float, default=3.0, help="summed confidence of fitting detections")
    ap.add_argument("--second_ratio", type=float, default=0.25, help="2nd object needs this fraction of the 1st")
    ap.add_argument("--third_ratio", type=float, default=0.6, help="a third object group needs this share of the top")
    ap.add_argument("--close_margin", type=float, default=0.2, help="confusable pair closer than this is flagged")
    ap.add_argument("--redecide", action="store_true", help="re-apply the rules to saved objects_id.json (no SAM 3)")
    ap.add_argument("--weights", default=str(ROOT / "weights/sam3"))
    ap.add_argument("--det_thresh", type=float, default=0.2, help="SAM 3 score threshold for this stage")
    a = ap.parse_args()
    mesh_ext = {}
    for nm in list(PROMPTS) + PAIR["wooden_board"]:
        mp = ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm / f"{nm}.obj"
        if mp.exists():
            mesh_ext[nm] = obj_extents(mp)
    mesh_ext["wooden_board"] = mesh_ext["wooden_piece_1"]
    seg = None
    if a.redecide:
        for r in a.runs:
            f = ROOT / "runs" / r / "objects_id.json"
            rec = json.loads(f.read_text())
            rec["objects"], rec["notes"] = decide(rec["support"], rec["board_pair_frames"], a)
            f.write_text(json.dumps(rec, indent=2))
            truth = json.loads((ROOT / "runs" / r / "episode.json").read_text()).get("objects")
            ok = "" if truth is None else ("  OK" if sorted(truth) == rec["objects"] else f"  WRONG (truth {sorted(truth)})")
            print(f"{r}: {rec['objects']}{'  AMBIGUOUS ' + str(rec['notes']) if rec['notes'] else ''}{ok}", flush=True)
        return
    for r in a.runs:
        run = ROOT / "runs" / r
        video = run / "input_undistorted.mp4"
        ki = json.loads((run / "intrinsics_stable.json").read_text())
        K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
        n = len(list((run / "depth_stereo").glob("*.png")))
        idx = np.linspace(0, n - 1, a.frames).round().astype(int)
        frames = frames_from_video(video, idx)
        if seg is None:
            seg = Sam3Segmenter(a.weights, run, n, score_thresh=a.det_thresh)
        support = {nm: 0.0 for nm in PROMPTS}
        hits = {nm: [] for nm in PROMPTS}
        pair_frames = 0
        for t, rgb in frames.items():
            depth = decode_depth(run / f"depth_stereo/{t:06d}.png")
            dets = seg.detect(rgb, PROMPTS)
            # Every fitting detection, under every label that produced it ...
            cand = []
            for name, conf, box, m in dets:
                g = geometry_check(m, depth, K, mesh_ext[name])
                if not (g["ok"] and g["reason"] != "no_depth" and g["fit"] >= 0.3):
                    continue
                if g["center"] is not None and np.linalg.norm(g["center"]) > 1.2:
                    continue  # the wearer handles objects within ~1.2 m
                # Size is a gate, not the label: it rejects gross mismatches (a cup cannot be the
                # pitcher, fit ~0.005), but among plausible sizes it cannot choose -- a thin white
                # pitcher shell measures pot-sized in stereo (pot fit ~1.0, pitcher ~0.44) while
                # SAM 3 itself is far surer it is a pitcher (0.93 vs 0.3-0.6). Appearance decides.
                cand.append({"name": name, "conf": conf, "fit": g["fit"], "box": box, "m": m,
                             "center": g["center"], "score": conf})
            # ... then one label per physical instance: several prompts fire on the same
            # object (a clear pitcher also answers to "white saucepan"); the mask goes to
            # the label that explains it best, so each instance votes exactly once.
            cand.sort(key=lambda c: -c["score"])
            inst = []
            for c in cand:
                same = [i for i in inst if mask_iou(c["m"], i["m"]) > 0.5]
                if not same:
                    inst.append(c)
            centers = {}
            for c in inst:
                centers.setdefault(c["name"], []).append(c["center"])
            for nm in PROMPTS:
                fitting = [(c["conf"], c["fit"], c["box"], c["m"]) for c in inst if c["name"] == nm]
                if nm == "white_pot_lid" and fitting:
                    # a lid resting on its pot all episode is scenery, not a manipulated object:
                    # only frames where the lid is clearly apart from the pot count
                    pots = centers.get("white_pot", [])
                    lc = [c["center"] for c in inst if c["name"] == nm][0]
                    if pots and min(np.linalg.norm(np.subtract(lc, pc)) for pc in pots) < 0.12:
                        fitting = []
                if fitting:
                    best = max(fitting, key=lambda x: x[0])
                    support[nm] += best[0]
                    hits[nm].append({"frame": t, "conf": round(best[0], 3), "fit": round(best[1], 3),
                                     "n_instances": len(fitting)})
                    if nm == "wooden_board" and len(fitting) >= 2:
                        pair_frames += 1
        objects, notes = decide(support, pair_frames, a)
        rec = {"objects": objects, "notes": notes, "support": {k: round(v, 3) for k, v in support.items()},
               "board_pair_frames": pair_frames, "frames": [int(t) for t in frames], "hits": hits}
        (run / "objects_id.json").write_text(json.dumps(rec, indent=2))
        truth = json.loads((run / "episode.json").read_text()).get("objects")
        verdict = "" if truth is None else ("  OK" if sorted(truth) == objects else f"  WRONG (truth {sorted(truth)})")
        print(f"{r}: {objects}  support={ {k: round(v, 1) for k, v in support.items() if v > 0} }{verdict}", flush=True)


if __name__ == "__main__":
    main()
