"""Stage 1: resolve and verify the egocentric camera calibration.

The ego rig is mounted rotated 180° (rotation_deg=180) but the stored videos are
already upright, so it is ambiguous whether the published intrinsics/extrinsics are
in the sensor frame or the stored-image frame. For each hypothesis we measure the
epipolar distance of SIFT matches between cameras (c<->b stereo, a<->c cross-rig)
using the calibrated fundamental matrix; the right hypothesis gives ~1 px.

Outputs:
  $V2D_ROOT/calib/ego_calib_resolved.json   (K, dist, extrinsics for the stored frames)
  $V2D_VIZ/01_calibration/report.json, epipolar/undistort images, rectified-stereo videos
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ["V2D_ROOT"])
DATA = Path(os.environ["V2D_DATA"])
OUT = Path(os.environ["V2D_VIZ"]) / "01_calibration"
CAL = json.loads((DATA / "public/meta/camera_calibration.json").read_text())
RZ = np.diag([-1.0, -1.0, 1.0, 1.0])  # 180° about the optical axis
EPISODES = [0, 12, 23, 41]
N_FRAMES = 8


def camera(name, rotate):
    c = CAL["cameras"][name]
    w, h = c["resolution"]
    K = np.array(c["intrinsic_matrix"], float)
    d = np.array(c["distortion_coefficients"], float)
    if rotate:  # x->-x, y->-y about the image centre
        K[0, 2], K[1, 2] = (w - 1) - K[0, 2], (h - 1) - K[1, 2]
        d[[2, 3, 8, 9, 10, 11]] *= -1  # tangential p1,p2 and thin-prism s1..s4 are odd
    return {"K": K, "dist": d, "size": (w, h)}


def extr(T, rotate):
    T = np.array(T, float)
    return RZ @ T @ RZ if rotate else T


def frame(split, ep, stream, idx):
    cap = cv2.VideoCapture(str(DATA / f"{split}/videos/chunk-000/observation.images.{stream}/episode_{ep:06d}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def matches(img1, img2):
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.7 * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def epipolar_error(p1, p2, cam1, cam2, T_1to2):
    """Symmetric epipolar distance (px, in undistorted pinhole coords)."""
    u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), cam1["K"], cam1["dist"], P=cam1["K"]).reshape(-1, 2)
    u2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), cam2["K"], cam2["dist"], P=cam2["K"]).reshape(-1, 2)
    R, t = T_1to2[:3, :3], T_1to2[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(cam2["K"]).T @ tx @ R @ np.linalg.inv(cam1["K"])
    h1, h2 = np.c_[u1, np.ones(len(u1))], np.c_[u2, np.ones(len(u2))]
    l2, l1 = h1 @ F.T, h2 @ F
    num = np.abs(np.sum(h2 * l2, 1))
    d = 0.5 * (num / np.hypot(l2[:, 0], l2[:, 1]) + num / np.hypot(l1[:, 0], l1[:, 1]))
    return d, u1, u2


def evaluate(rotate, samples):
    cams = {n: camera(n, rotate) for n in ("ego_cam_a", "ego_cam_b", "ego_cam_c")}
    pair = next(p for p in CAL["stereo_pairs"] if p["camera"] == "ego")
    T_c2b = extr(pair["left_to_right"], rotate)
    T_a2c = extr(CAL["cameras"]["ego_cam_a"]["extrinsics_to_stereo_left"], rotate)
    res = {"stereo_c_to_b": [], "cross_a_to_c": [], "disparity_negative_frac": []}
    for imgs in samples:
        p1, p2 = matches(imgs["c"], imgs["b"])
        d, u1, u2 = epipolar_error(p1, p2, cams["ego_cam_c"], cams["ego_cam_b"], T_c2b)
        res["stereo_c_to_b"] += d.tolist()
        res["disparity_negative_frac"].append(float(np.mean(u1[:, 0] - u2[:, 0] < 0)))
        p1, p2 = matches(imgs["a"], imgs["c"])
        d, _, _ = epipolar_error(p1, p2, cams["ego_cam_a"], cams["ego_cam_c"], T_a2c)
        res["cross_a_to_c"] += d.tolist()
    summary = {k: {"median_px": float(np.median(v)), "p90_px": float(np.percentile(v, 90)), "n": len(v)}
               for k, v in res.items() if k != "disparity_negative_frac"}
    summary["disparity_negative_frac"] = float(np.mean(res["disparity_negative_frac"]))
    return summary, cams, T_c2b, T_a2c


def square_pixel_K(cam):
    """Pinhole K for undistortion with square pixels and a centred principal point,
    as large a field of view as possible without invalid (black) border pixels.

    cv2.getOptimalNewCameraMatrix(alpha=0) fills the frame by scaling x and y
    differently (fx/fy differ by ~10% here), which breaks models that assume a
    single focal length (e.g. HaWoR).
    """
    w, h = cam["size"]
    t = np.linspace(0, 1, 400)
    edges = np.concatenate([np.c_[t * (w - 1), np.zeros_like(t)], np.c_[t * (w - 1), np.full_like(t, h - 1)],
                            np.c_[np.zeros_like(t), t * (h - 1)], np.c_[np.full_like(t, w - 1), t * (h - 1)]])
    norm = cv2.undistortPoints(edges.reshape(-1, 1, 2).astype(np.float64), cam["K"], cam["dist"]).reshape(-1, 2)
    aspect = (w - 1) / (h - 1)
    # Largest centred rectangle [-s*aspect, s*aspect] x [-s, s] inside the boundary.
    s = np.min(np.maximum(np.abs(norm[:, 0]) / aspect, np.abs(norm[:, 1])))
    f = (h - 1) / 2 / s
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])


def rectified_video(cams, T_c2b, ep):
    cl, cr = cams["ego_cam_c"], cams["ego_cam_b"]
    size = cl["size"]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cl["K"], cl["dist"], cr["K"], cr["dist"], size,
                                                 T_c2b[:3, :3], T_c2b[:3, 3], alpha=0)
    m1 = cv2.initUndistortRectifyMap(cl["K"], cl["dist"], R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(cr["K"], cr["dist"], R2, P2, size, cv2.CV_32FC1)
    base = DATA / "public/videos/chunk-000"
    capl = cv2.VideoCapture(str(base / f"observation.images.ego_cam_c/episode_{ep:06d}.mp4"))
    capr = cv2.VideoCapture(str(base / f"observation.images.ego_cam_b/episode_{ep:06d}.mp4"))
    fps = json.loads((DATA / "public/meta/info.json").read_text())["fps"]
    w, h = size
    out = cv2.VideoWriter(str(OUT / f"stereo_rectified_ep{ep:06d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h // 2))
    while True:
        okl, fl = capl.read()
        okr, fr = capr.read()
        if not (okl and okr):
            break
        rl, rr = cv2.remap(fl, *m1, cv2.INTER_LINEAR), cv2.remap(fr, *m2, cv2.INTER_LINEAR)
        both = np.hstack([rl, rr])
        for y in range(40, h, 60):
            cv2.line(both, (0, y), (2 * w, y), (0, 255, 0), 1)
        cv2.putText(both, "rectified L=ego_cam_c | R=ego_cam_b  (green = epipolar lines)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        out.write(cv2.resize(both, (w, h // 2), interpolation=cv2.INTER_AREA))
    out.release()
    return R1, R2, P1, P2, Q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for ep in EPISODES:
        n = int(cv2.VideoCapture(str(DATA / f"public/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(10, n - 10, N_FRAMES).astype(int):
            samples.append({k: frame("public", ep, f"ego_cam_{k}", idx) for k in "abc"})

    report = {}
    for name, rotate in (("as_published", False), ("rotated_180", True)):
        report[name], *_ = evaluate(rotate, samples)
        print(name, json.dumps(report[name]), flush=True)
    best = min(report, key=lambda k: report[k]["stereo_c_to_b"]["median_px"] + report[k]["cross_a_to_c"]["median_px"])
    report["chosen"] = best
    _, cams, T_c2b, T_a2c = evaluate(best == "rotated_180", samples[:1])
    print("chosen:", best, flush=True)

    R1, R2, P1, P2, Q = None, None, None, None, None
    for ep in (0, 23):
        R1, R2, P1, P2, Q = rectified_video(cams, T_c2b, ep)

    # Undistorted colour frame for a visual straight-line check.
    ca = cams["ego_cam_a"]
    Knew = square_pixel_K(ca)
    cap = cv2.VideoCapture(str(DATA / "public/videos/chunk-000/observation.images.ego_cam_a/episode_000000.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    _, img = cap.read()
    und = cv2.undistort(img, ca["K"], ca["dist"], None, Knew)
    cv2.imwrite(str(OUT / "ego_cam_a_raw_vs_undistorted.jpg"),
                cv2.resize(np.hstack([img, und]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    resolved = {
        "frame_convention": "stored (upright) video frames",
        "hypothesis": best, "verification": report,
        "cameras": {n: {"K": c["K"].tolist(), "dist": c["dist"].tolist(), "size": list(c["size"]),
                        "dist_model": "opencv_rational_14"} for n, c in cams.items()},
        "stereo": {"left": "ego_cam_c", "right": "ego_cam_b", "T_left_to_right": T_c2b.tolist(),
                   "R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist(),
                   "baseline_m": float(np.linalg.norm(T_c2b[:3, 3]))},
        "T_ego_cam_a_to_ego_cam_c": T_a2c.tolist(),
        "undistorted_ego_cam_a_K": Knew.tolist(),
    }
    (ROOT / "calib").mkdir(exist_ok=True)
    (ROOT / "calib/ego_calib_resolved.json").write_text(json.dumps(resolved, indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
