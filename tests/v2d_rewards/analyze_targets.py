"""Contact-location target statistics on a real reference (retargeted Sharpa frames as link positions, as in
training but without the scene FK): coverage, switch rate, link-to-target distance, per assignment rule.
Usage: python analyze_targets.py <parquet_dir> [control_fps=20] [motion_speed=0.5]"""
import sys

import numpy as np

from flash_chord.data import load_reference
from flash_chord.embodiments.sharpa_collision import sharpa_object_contact_links
from flash_chord.objectives.v2d_reference import SHARPA_MANO_SLOTS, assign_link_contact_targets

fps = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
speed = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
ref = load_reference(sys.argv[1], control_fps=fps, motion_speed=speed)
print("frames", ref.num_frames, "fps", ref.fps)
links = sharpa_object_contact_links()
for side in ("right", "left"):
    names = list(ref.frame_names(side))
    idx = [names.index(f"{side}_{n}") for n in links]
    link_w = np.asarray(ref.frame_pos_w(side))[:, idx]
    cp, cn = np.asarray(ref.contact_pos_w(side)), np.asarray(ref.contact_normal_w(side))
    cb = np.asarray(ref.contact_part_ids(side)) - 1
    in_contact = np.any(cb >= 0, 1)
    raw = cb >= 0
    raw_switch = np.sum(raw[1:] != raw[:-1]) / max(raw.sum(), 1)
    print(f"\n{side}: {in_contact.sum()} / {ref.num_frames} frames with human contact; raw MANO slot on/off switch rate {raw_switch:.3f}")
    for mode, kwargs in (
        ("nearest (no smoothing)", dict(assignment="nearest", hysteresis_margin=0.0, max_gap=0, min_segment=1, smoothing_window=1)),
        ("nearest", dict(assignment="nearest")),
        ("semantic", dict(assignment="semantic", semantic_slots=np.array([SHARPA_MANO_SLOTS[n] for n in links]))),
    ):
        t = assign_link_contact_targets(link_w, cp, cn, cb, np.asarray(ref.object_body_pos_w()), np.asarray(ref.object_body_quat_w()), **kwargs)
        cov = np.mean(t.body[in_contact] >= 0) if in_contact.any() else 0
        per_link = {n: int((t.body[:, j] >= 0).sum()) for j, n in enumerate(links)}
        frames_with = np.any(t.body >= 0, 1)
        jumps = np.linalg.norm(np.diff(t.pos_o, axis=0), axis=-1)[(t.body[1:] >= 0) & (t.body[:-1] >= 0)]
        print(f"  [{mode}] link coverage in contact frames {cov:.3f}; frames with >=1 target {frames_with.sum()} "
              f"(contact frames covered {np.mean(frames_with[in_contact]) if in_contact.any() else 0:.3f}); "
              f"switch rate {t.switch_rate():.3f}; ref link->target dist median {np.nanmedian(t.reference_distance) * 1e3:.1f} mm; "
              f"target jump p95 {np.percentile(jumps, 95) * 1e3 if jumps.size else 0:.2f} mm")
        print("     targeted frames per link:", per_link)
