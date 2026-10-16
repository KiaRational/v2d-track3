"""Why does the sim reference score far below our reconstruction against mocap? Alignment-free check per slot:
displacement from the first pose |p_t - p_0| and rotation angle from the first pose, for (a) the mocap reference,
(b) the policy's tracked reference (trajectory parquet reference_* columns, world 0), over frame_index; plus the best
time offset/scale of (b) onto (a), and the quaternion order that makes (b)'s rotation agree with (a).
Usage: python diag_ref_timing.py <fc_name> <episode>   (rg image)"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/scratch/kghasemz/v2d")
KIT = ROOT / "submission_kit/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb.submission_parquet import read_trajectory  # noqa: E402
from v2dlb.trajectory import episode_parquet, read_episode  # noqa: E402

fc, ep = sys.argv[1], int(sys.argv[2])
link = ROOT / "robot/score_links" / fc / f"episode_{ep:06d}.parquet"
tr = read_trajectory(link)
ref = read_episode(episode_parquet(ROOT / "data/track_3/public", ep), ep)
w0 = tr.world_index == tr.world_index.min()
print(f"rollout frames {tr.frame_index[w0].min()}..{tr.frame_index[w0].max()} n={len(np.unique(tr.frame_index[w0]))}; "
      f"mocap frames {ref.frame_index.min()}..{ref.frame_index.max()} n={len(ref.frame_index)}")
for s, name in enumerate(ref.object_names):
    m = w0 & (tr.object_slot == s)
    o = np.argsort(tr.frame_index[m])
    f_sim, P = tr.frame_index[m][o], tr.reference_xyzw[m][o]
    G = ref.pose_xyzw[:, s]
    dg = np.linalg.norm(G[:, :3] - G[0, :3], axis=1)
    ds = np.linalg.norm(P[:, :3] - P[0, :3], axis=1)
    rg = (Rotation.from_quat(G[:, 3:]) * Rotation.from_quat(G[0, 3:]).inv()).magnitude()
    rs_xyzw = (Rotation.from_quat(P[:, 3:]) * Rotation.from_quat(P[0, 3:]).inv()).magnitude()
    q = P[:, [4, 5, 6, 3]]  # if the stored order were wxyz
    rs_wxyz = (Rotation.from_quat(q) * Rotation.from_quat(q[0]).inv()).magnitude()
    print(f"\n[{s}] {name}: max displacement mocap {dg.max()*100:.1f} cm, sim ref {ds.max()*100:.1f} cm; "
          f"max rotation mocap {np.degrees(rg.max()):.0f} deg, sim ref (xyzw) {np.degrees(rs_xyzw.max()):.0f}, "
          f"(as wxyz) {np.degrees(rs_wxyz.max()):.0f}")
    print(f"  displacement-onset frame (>2 cm): mocap {ref.frame_index[np.argmax(dg > 0.02)] if (dg > 0.02).any() else None}, "
          f"sim ref {f_sim[np.argmax(ds > 0.02)] if (ds > 0.02).any() else None}")
    best = None
    for scale in np.linspace(0.85, 1.15, 31):
        for off in range(-40, 41):
            t = (ref.frame_index - ref.frame_index[0]) * scale + f_sim[0] + off
            ok = (t >= f_sim[0]) & (t <= f_sim[-1])
            if ok.sum() < 0.6 * len(t):
                continue
            e = np.mean(np.abs(np.interp(t[ok], f_sim, ds) - dg[ok]))
            if best is None or e < best[0]:
                best = (e, scale, off)
    t0 = np.isin(ref.frame_index, f_sim)
    e0 = np.mean(np.abs(np.interp(ref.frame_index[t0], f_sim, ds) - dg[t0]))
    print(f"  displacement-profile error as indexed: {e0*100:.2f} cm; best time map scale {best[1]:.3f} offset {best[2]} "
          f"-> {best[0]*100:.2f} cm")
    for k in range(0, len(ref.frame_index), max(1, len(ref.frame_index) // 12)):
        f = ref.frame_index[k]
        j = np.searchsorted(f_sim, f)
        sv = ds[j] * 100 if j < len(f_sim) and f_sim[j] == f else float("nan")
        print(f"   frame {f:4d}: mocap disp {dg[k]*100:5.1f} cm  sim ref disp {sv:5.1f} cm")
