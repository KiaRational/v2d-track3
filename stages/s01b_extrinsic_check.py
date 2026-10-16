"""Stage 1b: diagnose the ego_cam_a -> stereo extrinsic (published one gives ~19 px
epipolar error while the stereo pair gives ~2 px).

Tests interpretations of the published 4x4 (as-is / inverted, relative to cam_c or
cam_b, 180°-conjugated) and estimates the rotation + translation direction from
the images (essential matrix on undistorted SIFT matches, intrinsics known).
The best model is written back into ego_calib_resolved.json.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from s01_calibration import CAL, EPISODES, N_FRAMES, RZ, camera, epipolar_error, frame, matches, DATA

ROOT = Path(os.environ["V2D_ROOT"])
OUT = Path(os.environ["V2D_VIZ"]) / "01_calibration"


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    cams = {n: camera(n, False) for n in ("ego_cam_a", "ego_cam_b", "ego_cam_c")}
    T = np.array(CAL["cameras"]["ego_cam_a"]["extrinsics_to_stereo_left"], float)
    pair = next(p for p in CAL["stereo_pairs"] if p["camera"] == "ego")
    T_c2b = np.array(pair["left_to_right"], float)

    samples = []
    for ep in EPISODES:
        n = int(cv2.VideoCapture(str(DATA / f"public/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(10, n - 10, N_FRAMES).astype(int):
            samples.append({k: frame("public", ep, f"ego_cam_{k}", idx) for k in "abc"})
    m_ac = [matches(s["a"], s["c"]) for s in samples]
    m_ab = [matches(s["a"], s["b"]) for s in samples]

    def score(T_a2x, target):
        ms = m_ac if target == "c" else m_ab
        d = np.concatenate([epipolar_error(p1, p2, cams["ego_cam_a"], cams[f"ego_cam_{target}"], T_a2x)[0] for p1, p2 in ms])
        return float(np.median(d)), float(np.percentile(d, 90))

    inv = np.linalg.inv
    hyps = {
        "T as a->c": (T, "c"), "inv(T) as a->c": (inv(T), "c"),
        "T as a->b": (T, "b"), "inv(T) as a->b": (inv(T), "b"),
        "RZ T RZ as a->c": (RZ @ T @ RZ, "c"), "RZ inv(T) RZ as a->c": (RZ @ inv(T) @ RZ, "c"),
        "T then c->b, as a->b": (T_c2b @ T, "b"),
        # Factory a->b re-referenced to the relabeled stereo-left (cam_c).
        "inv(c->b) T as a->c": (inv(T_c2b) @ T, "c"),
    }
    results = {k: score(*v) for k, v in hyps.items()}

    # Direct estimate: essential matrix a->c from all matches (pixel coords undistorted).
    ua, uc = [], []
    for p1, p2 in m_ac:
        ua.append(cv2.undistortPoints(p1.reshape(-1, 1, 2), cams["ego_cam_a"]["K"], cams["ego_cam_a"]["dist"]).reshape(-1, 2))
        uc.append(cv2.undistortPoints(p2.reshape(-1, 1, 2), cams["ego_cam_c"]["K"], cams["ego_cam_c"]["dist"]).reshape(-1, 2))
    ua, uc = np.concatenate(ua), np.concatenate(uc)
    E, inl = cv2.findEssentialMat(ua, uc, np.eye(3), cv2.RANSAC, 0.999, 1.0 / 800)
    _, R_est, t_est, _ = cv2.recoverPose(E, ua, uc, np.eye(3), mask=inl)
    baseline = float(np.linalg.norm(T[:3, 3]))
    T_est = np.eye(4)
    T_est[:3, :3], T_est[:3, 3] = R_est, t_est.ravel() * baseline
    results["estimated (E-matrix, |t| from published)"] = score(T_est, "c")

    for k, (med, p90) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"{med:7.2f} px (p90 {p90:6.2f})  {k}", flush=True)
    print("published R vs estimated R: %.2f deg" % rot_deg(T[:3, :3].T @ R_est))
    print("published t dir:", np.round(T[:3, 3] / baseline, 3), " estimated t dir:", np.round(t_est.ravel(), 3))
    print("E-matrix inliers: %d / %d" % (int(inl.sum()), len(inl)))

    # Prefer the metric factory calibration unless the free E-matrix fit (which
    # absorbs sync/matching noise from only ~15% inliers) is clearly better.
    factory = "inv(c->b) T as a->c"
    best = min(results, key=lambda k: results[k][0])
    if best.startswith("estimated") and results[factory][0] < 1.5 * results[best][0]:
        best = factory
    if best.startswith("estimated"):
        T_best = T_est
    else:
        T_best, target = hyps[best]
        if target == "b":
            T_best = inv(T_c2b) @ T_best  # express as a->c for downstream use
    resolved_path = ROOT / "calib/ego_calib_resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["T_ego_cam_a_to_ego_cam_c"] = T_best.tolist()
    resolved["verification"]["cam_a_extrinsic"] = {"chosen": best, "results_median_p90_px": results}
    resolved_path.write_text(json.dumps(resolved, indent=2))
    (OUT / "cam_a_extrinsic_report.json").write_text(json.dumps(results, indent=2))
    print("chosen:", best)


if __name__ == "__main__":
    main()
