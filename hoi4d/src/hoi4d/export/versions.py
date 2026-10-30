"""Trajectory versions (README: preserve the initial reconstruction and every accepted revision, reversible).

hoi4d/versions.json: {name: {npz, parent, objects_variant, hands_variant, accepted, created, change}}
Each version is also written in the existing pipeline layout so every existing checker / renderer / exporter
runs on it unchanged:
  objects  runs/<r>/hoi4d_<name>/<obj>/poses/NNNNNN.json   {R, T} camera-from-object (colour camera a)
  hands    runs/<r>/stereo/hoi4d_<name>/<side>/NNNNNN.json same schema as mano_fit (mano, is_right, T, joints, flags)
The initial version points at the existing variants (no copy).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .. import io as hio
from ..geometry.assets import inv

ROOT = hio.ROOT


def reg_path(run):
    return hio.out_dir(run) / "versions.json"


def load(run) -> dict:
    p = reg_path(run)
    return json.loads(p.read_text()) if p.exists() else {}


def save(run, reg):
    hio.write_json(reg_path(run), reg)


def register_initial(run, tr, objects_variant, hands_variant):
    reg = load(run)
    p = hio.out_dir(run) / "trajectories_initial.npz"
    tr.save(p)
    reg["initial"] = {"npz": str(p), "parent": None, "objects_variant": objects_variant,
                      "hands_variant": hands_variant, "accepted": True, "created": hio.now(),
                      "change": "existing pipeline output (adapter)"}
    reg.pop("_current", None)  # a fresh ingest restarts from the initial version (older versions kept for provenance)
    save(run, reg)
    return reg


def write_variant(run: str, name: str, tr, parent: str, change: dict, base_hands_variant: str | None) -> dict:
    """New version: npz + pipeline-layout variant dirs. Not accepted until stage-10 validation says so."""
    r = ROOT / "runs" / run
    T_wc = tr.extra["T_world_cam"]
    tree = os.environ.get("HOI4D_DIR", "hoi4d")   # a separate tree (e.g. hoi4d_c) gets its own variant dirs
    ov = f"{tree}_{name}"
    od = r / ov
    if od.exists():
        shutil.rmtree(od)
    for n, o in tr.objects.items():
        d = od / n / "poses"
        d.mkdir(parents=True, exist_ok=True)
        for f in np.where(o["valid"])[0]:
            Tc = inv(T_wc[f]) @ o["T_world_obj"][f]
            (d / f"{f:06d}.json").write_text(json.dumps({"R": Tc[:3, :3].tolist(), "T": Tc[:3, 3].tolist()}))
    hv = None
    if tr.hands:
        hv = ov
        hd_root = r / "stereo" / hv
        if hd_root.exists():
            shutil.rmtree(hd_root)
        for h, hd in tr.hands.items():
            side = h.split("_")[1]
            d = hd_root / side
            d.mkdir(parents=True, exist_ok=True)
            src_dir = r / "stereo" / (base_hands_variant or "") / side
            for f in np.where(hd["valid"])[0]:
                base = json.loads((src_dir / f"{f:06d}.json").read_text()) if (src_dir / f"{f:06d}.json").exists() else {}
                Twc = T_wc[f]
                Jc = (hd["joints"][f] - Twc[:3, 3]) @ Twc[:3, :3]
                rec = {**base, "mano": {"betas": hd["betas"].tolist(), "global_orient": hd["global_orient"][f].tolist(),
                                        "hand_pose": hd["hand_pose"][f].tolist()},
                       "is_right": side == "right", "T": hd["transl"][f].tolist(), "joints": Jc.tolist()}
                (d / f"{f:06d}.json").write_text(json.dumps(rec))
        # keep the report the export/hand tools expect (variant consistency check)
        rp = r / "stereo" / (base_hands_variant or "") / "report.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            rep.setdefault("args", {})["variant"] = ov
            (hd_root / "report.json").write_text(json.dumps(rep))
    p = hio.out_dir(run) / f"trajectories_{name}.npz"
    tr.meta.update({"revision": name, "parent": parent, "refinement": change, "created": hio.now()})
    tr.save(p)
    reg = load(run)
    reg[name] = {"npz": str(p), "parent": parent, "objects_variant": ov, "hands_variant": hv or reg[parent]["hands_variant"],
                 "accepted": False, "created": hio.now(), "change": change}
    save(run, reg)
    return reg[name]


def set_accepted(run, name, accepted: bool, decision: dict):
    reg = load(run)
    reg[name]["accepted"] = accepted
    reg[name]["decision"] = decision
    if accepted:
        reg["_current"] = name
    save(run, reg)


def current(run) -> str:
    return load(run).get("_current", "initial")


def reject_cleanup(run, name):
    """A rejected version keeps its npz + registry entry (alternatives are preserved, README stage 10) but its
    pipeline-layout pose dirs are removed (they are regenerable from the npz)."""
    r = ROOT / "runs" / run
    reg = load(run)
    v = reg.get(name, {})
    for d in (r / str(v.get("objects_variant")), r / "stereo" / str(v.get("hands_variant"))):
        if d.name.startswith("hoi4d_") and d.is_dir():
            shutil.rmtree(d)
