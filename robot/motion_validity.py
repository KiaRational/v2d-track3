"""Motion-validity pass on a training reference: no object may teleport.

Found on 6/40 references (scan 2026-10-23): the spoon flips 90-170 deg for one or two frames and flips back
(31-60 rad/s: the pose estimator swapping the near-symmetric spoon's bowl up / bowl down), a pitcher / sweeper jumps
once. A policy cannot make an object follow such a target -- the tracking error explodes and the training episode
terminates on the object, so these frames stop the policy from ever learning the rest of the motion.

Per object: steps faster than a hand-held object moves (angular > W_MAX rad/s or linear > V_MAX m/s) are grouped
into bursts (suspect steps <= MERGE frames apart). A burst whose pose after it is close to the pose before it
(jump-and-return: < RETURN_DEG / RETURN_M) is a detection error: its frames are replaced by the slerp / lerp between
the last frame before and the first frame after the burst. One-way fast moves (no return) are left alone -- they may
be real. Writes the parquet in place with a provenance note in the schema metadata.
Usage: python motion_validity.py <hmd_root> <seq>[,<seq>...]   (pyarrow + scipy; any image)
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot, Slerp

W_MAX, V_MAX, MERGE, MAX_BURST, RETURN_DEG, RETURN_M = 12.0, 1.5, 6, 15, 35.0, 0.04


def repair(P, Q, fps):
    """P (F,3), Q (F,4 xyzw) of one object -> repaired copies + report."""
    F = len(P)
    R = Rot.from_quat(Q)
    v = np.linalg.norm(np.diff(P, axis=0), axis=1) * fps
    w = (R[:-1].inv() * R[1:]).magnitude() * fps
    sus = np.where((v > V_MAX) | (w > W_MAX))[0]  # step t -> t+1
    P, Q = P.copy(), Q.copy()
    rep = []
    i = 0
    while i < len(sus):
        j = i
        while j + 1 < len(sus) and sus[j + 1] - sus[j] <= MERGE:
            j += 1
        a, b = int(sus[i]), int(sus[j]) + 1  # last good frame before, first frame after the burst
        span = b - a - 1
        if span <= MAX_BURST and b < F:
            dr = np.degrees((R[a].inv() * R[b]).magnitude())
            dp = np.linalg.norm(P[b] - P[a])
            if dr < RETURN_DEG and dp < RETURN_M:
                key = Slerp([0, 1], Rot.from_quat([Q[a], Q[b]]))
                for f in range(a + 1, b):
                    u = (f - a) / (b - a)
                    Q[f] = key(u).as_quat()
                    P[f] = (1 - u) * P[a] + u * P[b]
                rep.append({"frames": [a + 1, b - 1], "peak_rad_s": round(float(w[a:b].max()), 1),
                            "peak_m_s": round(float(v[a:b].max()), 2), "action": "replaced (jump-and-return)"})
            else:
                rep.append({"frames": [a + 1, b - 1], "peak_rad_s": round(float(w[a:b].max()), 1),
                            "action": f"kept (one-way: {dr:.0f} deg / {dp * 100:.1f} cm)"})
        i = j + 1
    return P, Q, rep


def run(hmd, seq):
    pqf = next((Path(hmd) / f"ego_recon/processed/sequence_id={seq}").rglob("*.parquet"))
    t = pq.read_table(pqf)
    fps = float(t.column("fps")[0].as_py())
    names = t.column("object_body_names")[0].as_py()
    pos = np.asarray(t.column("object_body_position")[0].as_py(), float)
    wxyz = np.asarray(t.column("object_body_wxyz")[0].as_py(), float)
    newP, newQ, report = pos.copy(), wxyz.copy(), {}
    for k, n in enumerate(names):
        P, Q, rep = repair(pos[:, k], wxyz[:, k][:, [1, 2, 3, 0]], fps)
        if rep:
            report[n] = rep
            newP[:, k], newQ[:, k] = P, Q[:, [3, 0, 1, 2]]
    print(f"{seq}: {report if report else 'clean'}", flush=True)
    if not any(r["action"].startswith("replaced") for rs in report.values() for r in rs):
        return
    cols = {"object_body_position": pa.array([newP.tolist()], type=t.schema.field("object_body_position").type),
            "object_body_wxyz": pa.array([newQ.tolist()], type=t.schema.field("object_body_wxyz").type)}
    root = np.asarray(t.column("object_root_position")[0].as_py(), float)
    if np.allclose(root, pos[:, 0], atol=1e-6):  # root == body 0: keep them consistent
        cols["object_root_position"] = pa.array([newP[:, 0].tolist()], type=t.schema.field("object_root_position").type)
        cols["object_root_axis_angle"] = pa.array([Rot.from_quat(newQ[:, 0][:, [1, 2, 3, 0]]).as_rotvec().tolist()],
                                                  type=t.schema.field("object_root_axis_angle").type)
    arrays = [cols.get(c, t.column(c)) for c in t.column_names]
    md = dict(t.schema.metadata or {}); md[b"v2d_motion_validity"] = str(report).encode()
    pq.write_table(pa.Table.from_arrays(arrays, schema=t.schema.with_metadata(md)), pqf)


if __name__ == "__main__":
    for s in sys.argv[2].split(","):
        run(sys.argv[1], s)
