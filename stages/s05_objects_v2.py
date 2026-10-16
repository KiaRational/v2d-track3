"""Stage 5 v2: robust multi-object segmentation/tracking -> FoundationPose + EKF.

Differences from s05_objects.py (v1), see objects_v2/sam2_driver.py for details:
  * several descriptive GroundingDINO prompts per object, run on every 5th frame;
  * candidate masks validated geometrically against the mesh size with stereo depth;
  * references chosen by a joint object<->candidate assignment (v1 re-claimed the
    same object for the second id because its de-dup only compared boxes within
    the same frame); same-size look-alikes are resolved by mesh TEXTURE (below);
  * SAM2 video tracking validated per frame (size, duplicates, identity) and
    re-prompted where it fails; frames still invalid get no mask.
Same-size look-alikes (the two wooden boards) get their identity from mesh TEXTURE
(markers/slot/labels) rather than shape, which also fixes the 180 deg flips.
Output layout is unchanged (<run>/<out>/masks/<id>/NNNNNN.png - now only for valid
frames -, <run>/<out>/<name>/poses[_smoothed], objects.json, viz), plus
<out>/masks_ekf/ (all frames, empty = invalid), <out>/tracking_report.json and
<out>/identity.json; an unresolvable look-alike pair is marked in objects.json with
"identity_ambiguous": true and "identity_alternatives".
Usage (GPU job, host env): python s05_objects_v2.py <run_name> [--objects a,b] [--out objects]
       [--segmenter sam2_gdino|sam3] [--sam3_weights /scratch/kghasemz/v2d/weights/sam3]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing  # noqa: E402
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
W = ROOT / "weights"
HERE = Path(__file__).resolve().parent / "objects_v2"
HOME_CODE = Path.home() / "v2d-vulcan"
OBJECT_IDS = [1, 4, 5, 6, 7]  # 2 and 3 are the hands

# Descriptive prompts: the bare names confused GroundingDINO ("a mini sweeper"
# boxed the dust pan, "a wooden piece 1" boxed the round table).
PROMPTS = {
    "white_pot": ["white saucepan with a long handle", "white pot"],
    "white_pot_lid": ["white round pot lid with a handle", "white lid"],
    "mini_sweeper": ["small hand brush with bristles and a wooden ring handle", "brush"],
    "mini_dust_pan": ["small yellow dustpan", "dustpan"],
    "wooden_piece_1": ["rectangular wooden board", "wooden plank"],
    "wooden_piece_2": ["rectangular wooden board", "wooden plank"],
    "plastic_dish_rack": ["white plastic dish rack", "dish drainer"],
    "blue_cup": ["blue cup", "blue mug"],
    "beige_cup": ["beige cup", "beige mug"],
    "water_pitcher": ["water pitcher", "plastic jug", "jug", "pitcher", "watering can",
                      "plastic container with handle"],
}


# SAM 3 wants short noun phrases ("concepts").
SAM3_PROMPTS = {
    "white_pot": ["white saucepan"], "white_pot_lid": ["white pot lid"],
    "mini_sweeper": ["hand brush"], "mini_dust_pan": ["yellow dustpan"],
    "wooden_piece_1": ["wooden board"], "wooden_piece_2": ["wooden board"],
    "plastic_dish_rack": ["dish rack"], "blue_cup": ["blue cup"], "beige_cup": ["beige cup"],
    "water_pitcher": ["water pitcher", "plastic jug", "jug", "pitcher", "watering can"],
    "wooden_spoon": ["wooden spoon"],  # SAM 3 score 0.9+ on the spoon episodes (tmp_diag/spoon)
}

def container(image, script, args):
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{ROOT}:{ROOT}", "-v", f"{HOME_CODE}:{HOME_CODE}",
           image, "python", str(HERE / script)] + [str(a) for a in args]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--objects", default=None)
    ap.add_argument("--out", default="objects")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--segmenter", choices=["sam2_gdino", "sam3"], default="sam2_gdino")
    ap.add_argument("--sam3_weights", default=str(W / "sam3"), help="local facebook/sam3 snapshot (gated)")
    ap.add_argument("--no_viz", action="store_true")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    meta = json.loads((run / "episode.json").read_text())
    names = args.objects.split(",") if args.objects else meta["objects"]
    recon = run / "recon_hawor"
    frames_dir = recon / "frames"
    n = len(list(frames_dir.glob("*.png")))
    work = run / args.out
    work.mkdir(exist_ok=True)

    # 1) GroundingDINO, several prompts per object, every `stride` frames (sam2_gdino only;
    #    the sam3 segmenter detects with its own text prompts inside the driver)
    prompts = {nm: PROMPTS.get(nm, ["a " + nm.replace("_", " ")]) for nm in names}
    (work / "gdino_prompts.json").write_text(json.dumps(prompts, indent=1))
    dets = work / "gdino_dets.json"
    if args.segmenter == "sam2_gdino" and not dets.exists():
        container("v2d_grounding_dino", "gdino_detect.py",
                  ["--frames_dir", frames_dir, "--frames", ",".join(map(str, range(0, n, args.stride))),
                   "--prompts", work / "gdino_prompts.json", "--model_dir", W / "grounding_dino", "--out", dets])

    # 2) SAM2 tracking with geometric validation and re-prompting
    objs = []
    for k, nm in enumerate(names):
        mesh = ROOT / "meshes_obj" / nm / f"{nm}.obj"
        o = {"id": OBJECT_IDS[k], "name": nm, "mesh": str(mesh),
             "mesh_ext": trimesh.load(mesh, force="mesh").extents.tolist()}
        objs.append(o)
    hand_prompts = [p for p in json.loads((recon / "sam2_prompts.json").read_text())["prompts"]
                    if p["object_id"] in (2, 3)]
    for p in hand_prompts:
        p["mask_path"] = str(recon / p["mask_path"])
    masks_dir, masks_ekf = work / "masks", work / "masks_ekf"
    report = work / "tracking_report.json"
    cfg = {"run_dir": str(run), "work_dir": str(work), "video": str(run / "input_undistorted.mp4"),
           "n_frames": n, "objects": [{k: v for k, v in o.items() if k != "mesh"} for o in objs],
           "dets": str(dets), "hand_prompts": hand_prompts, "weights_dir": str(W / "sam2"),
           "masks_dir": str(masks_dir), "masks_ekf_dir": str(masks_ekf), "report": str(report),
           "passes": args.passes, "stride": args.stride, "segmenter": args.segmenter,
           "sam3_weights_dir": args.sam3_weights, "text_prompts": {nm: SAM3_PROMPTS.get(nm, [nm.replace("_", " ")]) for nm in names}}
    (work / "sam2_driver_cfg.json").write_text(json.dumps(cfg, indent=1))
    if not report.exists():
        container("v2d_sam2" if args.segmenter == "sam2_gdino" else "sam3", "sam2_driver.py",
                  ["--cfg", work / "sam2_driver_cfg.json"])
    rep = json.loads(report.read_text())

    # 3) FoundationPose + EKF per object on metric stereo depth. Objects whose meshes
    #    are the same size (the two wooden boards) get their identity decided by
    #    texture afterwards, which needs poses - so FP may run twice for those.
    depth_dir = run / "depth_stereo" if (run / "depth_stereo").is_dir() else recon / "depth"
    intr = recon / "intrinsics_stable.json"

    def fit_poses():
        done = []
        for o in objs:
            if o["name"] not in rep.get("ref_frames", {}):
                print(f"WARNING: {o['name']} never detected; skipping", flush=True)
                continue
            have = sorted(int(p.stem) for p in (masks_dir / str(o["id"])).glob("*.png"))
            if not have:
                print(f"WARNING: {o['name']} has no valid mask; skipping", flush=True)
                continue
            r0 = rep["ref_frames"][o["name"]]
            ref = min(have, key=lambda t: abs(t - r0))
            o.update(prompt=prompts[o["name"]][0], ref_frame=ref, box=rep["ref_boxes"][o["name"]],
                     dino_confidence=rep["ref_conf"][o["name"]], n_valid_masks=len(have))
            od = work / o["name"]
            if not (od / "poses").is_dir() or len(list((od / "poses").glob("*.json"))) < n:
                run_video_to_poses(video_path=str(run / "input_undistorted.mp4"), depth_folder=str(depth_dir),
                                   masks_folder=str(masks_dir / str(o["id"])), camera_intrinsics_path=str(intr),
                                   mesh_path=o["mesh"], poses_dir=str(od / "poses"), weights_dir=str(W / "foundation_pose"),
                                   reference_frame=ref, mask_depth=True, reregister_iou_thresh=0.3,
                                   target_width=1014, target_height=760, dev=False)
            if not (od / "poses_smoothed").is_dir():
                run_ekf_smoothing(poses_dir=str(od / "poses"), mesh_path=o["mesh"], intrinsics_path=str(intr),
                                  weights_dir=str(W / "foundation_pose"), output_dir=str(od / "poses_smoothed"),
                                  masks_folder=str(masks_ekf / str(o["id"])), process_noise_xy=0.01, process_noise_z=0.01,
                                  process_noise_r=0.02, measurement_noise_xy=0.01, measurement_noise_z=0.04,
                                  measurement_noise_r=0.02, dev=False)
            o["poses"] = str(od / "poses_smoothed")
            o["depth_source"] = depth_dir.name
            done.append(o)
        return done

    done = fit_poses()

    # 4) identity of same-size look-alikes, by texture (shape/ICP cannot separate them)
    idf = work / "identity.json"
    if not idf.exists() and len(done) > 1:
        sys.path.insert(0, str(HERE))
        from objects_v2.identity_texture import resolve as resolve_identity
        idrep = resolve_identity(run, work, done, n)
        for r in idrep:
            if r.get("decision") == "swap" and r.get("confident"):
                a, b = r["pair"]
                for k in ("ref_frames", "ref_boxes", "ref_conf"):
                    if a in rep.get(k, {}) and b in rep.get(k, {}):
                        rep[k][a], rep[k][b] = rep[k][b], rep[k][a]
                print(f"identity: swapped {a} <-> {b} (margin {r['margin']:.3f})", flush=True)
        idf.write_text(json.dumps(idrep, indent=1))
        rep["identity"] = idrep
        report.write_text(json.dumps(rep, indent=1))
        if any(r.get("decision") == "swap" and r.get("confident") for r in idrep):
            done = fit_poses()          # poses were dropped; refit with the right meshes
    elif idf.exists():
        idrep = json.loads(idf.read_text())
    else:
        idrep = []

    # record any unresolved ambiguity so downstream can report both readings
    amb = {}
    for r in idrep:
        if r.get("decision") == "unknown" or (r.get("confident") is False):
            for nm in r["pair"]:
                amb[nm] = [x for x in r["pair"] if x != nm]
    for o in done:
        o.pop("mesh_ext", None)
        if o["name"] in amb:
            o["identity_ambiguous"] = True
            o["identity_alternatives"] = amb[o["name"]]

    (work / "objects.json").write_text(json.dumps(done, indent=2))
    if not args.no_viz and args.out == "objects":
        from s05_objects import render_viz
        K = np.array(json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())["undistorted_ego_cam_a_K"])
        render_viz(run, done, n, K, meta["fps"])
    print(f"done: {[o['name'] for o in done]}"
          + (f"  AMBIGUOUS: {sorted(amb)}" if amb else ""), flush=True)


if __name__ == "__main__":
    main()
