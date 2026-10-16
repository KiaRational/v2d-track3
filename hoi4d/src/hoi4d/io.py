"""Artifact read/write + validation for the README data contracts (no jsonschema dependency: the containers
differ, so required keys and array shapes are checked here directly; schemas/*.json document the same).

trajectories_*.npz  (one file per version; refined adds source revision + refinement metadata)
    t_s (F,)  frame (F,) int
    obj/<name>/T_world_obj (F,4,4)  obj/<name>/valid (F,) bool  obj/<name>/source (F,) uint8 (conventions.SOURCE_CODES)
    obj/<name>/sigma_t_m (F,)  obj/<name>/sigma_r_rad (F,)   (NaN when unknown)
    hand_<side>/global_orient (F,3) hand_pose (F,45) betas (10,) transl (F,3) joints (F,21,3) valid (F,) source (F,)
    meta_json  (json string: run, revision, parent, created, conventions, inputs, refinement)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from . import conventions as C

ROOT = Path("/scratch/kghasemz/v2d")


def out_dir(run: str) -> Path:
    d = ROOT / "runs" / run / "hoi4d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def write_json(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_np_default))
    tmp.replace(p)
    return p


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def append_jsonl(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"time": now(), **rec}, default=_np_default) + "\n")


# ---------------------------------------------------------------- trajectories
class Trajectories:
    """In-memory trajectory set; the single schema for initial and refined versions."""

    def __init__(self, t_s, frame, meta=None):
        self.t_s = np.asarray(t_s, float)
        self.frame = np.asarray(frame, int)
        self.objects: dict[str, dict[str, np.ndarray]] = {}
        self.hands: dict[str, dict[str, np.ndarray]] = {}
        self.meta = dict(meta or {})
        self.extra: dict[str, np.ndarray] = {}  # e.g. T_world_cam (F,4,4)

    @property
    def F(self):
        return len(self.t_s)

    def add_object(self, name, T, valid, source, sigma_t=None, sigma_r=None):
        F = self.F
        T = np.asarray(T, float).reshape(F, 4, 4).copy()
        valid = np.asarray(valid, bool).reshape(F)
        T[~valid] = np.nan
        self.objects[name] = {"T_world_obj": T, "valid": valid, "source": np.asarray(source, np.uint8).reshape(F),
                              "sigma_t_m": np.full(F, np.nan) if sigma_t is None else np.asarray(sigma_t, float),
                              "sigma_r_rad": np.full(F, np.nan) if sigma_r is None else np.asarray(sigma_r, float)}

    def add_hand(self, side, global_orient, hand_pose, betas, transl, joints, valid, source):
        F = self.F
        valid = np.asarray(valid, bool).reshape(F)
        d = {"global_orient": np.asarray(global_orient, float).reshape(F, 3).copy(),
             "hand_pose": np.asarray(hand_pose, float).reshape(F, 45).copy(),
             "betas": np.asarray(betas, float).reshape(-1)[:10].copy(),
             "transl": np.asarray(transl, float).reshape(F, 3).copy(),
             "joints": np.asarray(joints, float).reshape(F, 21, 3).copy(),
             "valid": valid, "source": np.asarray(source, np.uint8).reshape(F)}
        for k in ("global_orient", "hand_pose", "transl", "joints"):
            d[k][~valid] = np.nan
        self.hands[f"hand_{side}"] = d

    def copy(self):
        o = Trajectories(self.t_s.copy(), self.frame.copy(), json.loads(json.dumps(self.meta, default=_np_default)))
        o.objects = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in self.objects.items()}
        o.hands = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in self.hands.items()}
        o.extra = {k: v.copy() for k, v in self.extra.items()}
        return o

    def entities(self):
        return list(self.objects) + list(self.hands)

    def save(self, p: Path):
        validate_trajectories(self)
        arrs = {"t_s": self.t_s, "frame": self.frame}
        for n, d in self.objects.items():
            arrs.update({f"obj/{n}/{k}": v for k, v in d.items()})
        for n, d in self.hands.items():
            arrs.update({f"{n}/{k}": v for k, v in d.items()})
        arrs.update({f"extra/{k}": v for k, v in self.extra.items()})
        meta = {"conventions": {"quat": C.QUAT_ORDER, "units": C.UNITS, "transform": C.TRANSFORM_CONVENTION,
                                "source_codes": C.SOURCE_CODES, "missing": "NaN + valid=False"}, **self.meta}
        arrs["meta_json"] = np.array(json.dumps(meta, default=_np_default))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        tmp.replace(p)
        return p

    @staticmethod
    def load(p: Path) -> "Trajectories":
        z = np.load(p, allow_pickle=False)
        t = Trajectories(z["t_s"], z["frame"], json.loads(str(z["meta_json"])))
        for k in z.files:
            parts = k.split("/")
            if parts[0] == "obj":
                t.objects.setdefault(parts[1], {})[parts[2]] = z[k]
            elif parts[0].startswith("hand_"):
                t.hands.setdefault(parts[0], {})[parts[1]] = z[k]
            elif parts[0] == "extra":
                t.extra[parts[1]] = z[k]
        validate_trajectories(t)
        return t


def validate_trajectories(t: Trajectories):
    F = t.F
    assert t.frame.shape == (F,) and np.all(np.diff(t.t_s) > 0), "t_s must be strictly increasing"
    for n, d in t.objects.items():
        T, v = d["T_world_obj"], d["valid"]
        assert T.shape == (F, 4, 4) and v.shape == (F,), n
        assert np.all(np.isfinite(T[v])), f"{n}: valid frames with non-finite pose"
        assert np.all(np.isnan(T[~v])), f"{n}: invalid frames must be NaN, never a placeholder pose"
        R = T[v, :3, :3]
        if len(R):
            err = np.abs(np.einsum("fij,fkj->fik", R, R) - np.eye(3)).max()
            assert err < 1e-4, f"{n}: rotation not orthonormal ({err:.1e})"
            assert np.all(d["source"][v] > 0) and np.all(d["source"][~v] == 0), f"{n}: source codes vs valid"
    for n, d in t.hands.items():
        assert n in C.HAND_ENTITIES, n
        v = d["valid"]
        assert d["joints"].shape == (F, 21, 3) and d["hand_pose"].shape == (F, 45), n
        assert np.all(np.isfinite(d["joints"][v])) and np.all(np.isnan(d["joints"][~v])), n


# ---------------------------------------------------------------- json artifacts: required keys
REQUIRED = {
    "scene_manifest": ["run", "units", "quaternion_order", "transform_convention", "world_frame", "gravity_world",
                       "fps", "frame_count", "timestamps", "cameras", "support", "entities", "scale_sources"],
    "task": ["description", "entities", "phases", "contact_hypotheses", "alternatives", "unknowns", "model",
             "frames_used"],
    "contacts": ["version", "contacts"],
    "anomalies": ["version", "anomalies"],
    "metrics": ["version", "reconstruction", "tracking", "contact", "physics", "runtime"],
}
ITEM_REQUIRED = {
    "task.phases": ["name", "t_start_s", "t_end_s", "entities", "evidence_frames", "confidence"],
    "contacts.contacts": ["entity_a", "entity_b", "mode", "t_start_s", "t_end_s", "region", "evidence", "confidence"],
    "anomalies.anomalies": ["id", "type", "entities", "t_start_s", "t_end_s", "frames", "measurements", "thresholds",
                            "severity", "status", "evidence"],
}


def validate(kind: str, obj: dict) -> dict:
    miss = [k for k in REQUIRED[kind] if k not in obj]
    if miss:
        raise ValueError(f"{kind}: missing keys {miss}")
    for path, keys in ITEM_REQUIRED.items():
        art, field = path.split(".")
        if art != kind:
            continue
        for i, it in enumerate(obj[field]):
            m = [k for k in keys if k not in it]
            if m:
                raise ValueError(f"{kind}.{field}[{i}]: missing keys {m}")
    if kind == "anomalies":
        for a in obj["anomalies"]:
            if a["type"] not in C.ANOMALY_TYPES or a["status"] not in C.ANOMALY_STATUS:
                raise ValueError(f"anomaly {a['id']}: bad type/status {a['type']}/{a['status']}")
    if kind == "contacts":
        for c in obj["contacts"]:
            if c["mode"] not in C.CONTACT_MODES:
                raise ValueError(f"contact mode {c['mode']}")
    return obj


def write_artifact(run: str, kind: str, obj: dict, name: str | None = None) -> Path:
    validate(kind, obj)
    return write_json(out_dir(run) / (name or f"{kind}.json"), obj)
