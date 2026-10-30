"""Export a finished Dyn-HaMR world-frame fit as plain arrays (runs INSIDE the dynhamr container, cwd = dyn-hamr/).

Uses Dyn-HaMR's own MANO wrapper and run_mano (the exact forward its renderer uses: right-hand MANO, left hands
mirrored in x, joints in OpenPose order), no extra smoothing (the optimisation output as is).
Writes <out>.npz: verts (B,T,778,3), joints (B,T,21,3) in the Dyn-HaMR world (= our hoi4d world, cameras supplied),
is_right (B,T), vis_mask (B,T) = frames with a detection, track_ids, world_scale, cam_R/cam_t (w2c, for the check).
Usage: python dynhamr_export.py <result_dir> <out.npz>
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from body_model import MANO
from body_model.utils import run_mano
from optim.output import load_track_info


def main(res_dir, out):
    path = sorted(glob.glob(os.path.join(res_dir, "smooth_fit", "*_world_results.npz")))[-1]
    res = {k: torch.as_tensor(v).float() for k, v in np.load(path).items()}
    track_ids, vis, _, _ = load_track_info(os.path.join(res_dir, "track_info.json"))
    B, T = res["trans"].shape[:2]
    cfg = {"model_path": "../_DATA/data/mano", "gender": "neutral", "num_hand_joints": 15,
           "mean_params": "../_DATA/data/mano_mean_params.npz", "create_body_pose": False}
    bm = MANO(batch_size=B * T, pose2rot=True, **cfg)
    with torch.no_grad():
        o = run_mano(bm, res["trans"], res["root_orient"], res["pose_body"], res["is_right"], res.get("betas"))
    np.savez_compressed(out, verts=o["vertices"].numpy().astype(np.float32), joints=o["joints"].numpy().astype(np.float32),
                        is_right=o["is_right"].numpy(), vis_mask=vis.numpy().astype(bool)[:, :T], track_ids=track_ids.numpy(),
                        world_scale=res.get("world_scale", torch.ones(1)).numpy(), cam_R=res["cam_R"].numpy(), cam_t=res["cam_t"].numpy(),
                        source=path)
    print(json.dumps({"result": path, "tracks": int(B), "frames": int(T),
                      "is_right": o["is_right"][:, 0].tolist(), "visible": vis.sum(1).tolist(),
                      "world_scale": res.get("world_scale", torch.ones(1)).flatten()[:4].tolist()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
