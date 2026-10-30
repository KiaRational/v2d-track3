"""Robust multi-object SAM2 tracking with geometric validation and re-prompting
(runs INSIDE the v2d_sam2 container).

1. Candidates: every GroundingDINO box (all prompts, sampled frames) -> SAM2 image
   mask; boxes giving the same mask are merged, keeping per-object confidences.
   Each (candidate, object) gets a geometric fit: robust 3D extent of the mask
   (metric stereo depth) vs the object's mesh bounding box. Table-sized or
   speck-sized masks are invalid for that object.
2. Reference: the sampled frame whose optimal (Hungarian) object<->candidate
   assignment is best; objects missing there take their best frame elsewhere.
   (Same-size objects are disambiguated afterwards on the host, identity_icp.py.)
   (v1 bug: de-dup only compared boxes within the SAME frame, so the second object
   re-claimed the first object's box 10 frames later -> both ids tracked one object.)
3. Joint SAM2 video propagation (objects + both hands), then per-frame validation:
   geometry, duplicates (two ids on one mask), identity disagreement with the
   per-frame candidate assignment. Failing segments are re-prompted with the
   assigned candidate mask (continuity-weighted); repeat up to --passes times.
4. Final cleanup: depth-connected-component trimming; frames still invalid or
   duplicated are dropped (no mask file; an empty mask goes to masks_ekf/ so the
   EKF down-weights FoundationPose there).

Usage: python sam2_driver.py --cfg CFG.json   (see s05_objects_v2.py for the keys)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import (clean_mask, containment, decode_depth, geometry_check, load_mask,  # noqa: E402
                  mask_iou)

import cv2  # noqa: E402

HAND_IDS = (2, 3)
MERGE_IOU = 0.7      # candidate masks this similar are the same physical thing
DUP_IOU = 0.6        # two object tracks this similar = duplicate
DUP_FIT_MARGIN = 0.15  # mesh-size fit gap that resolves a duplicate in one id's favour
REF_MIN_FIT = 0.25   # a reference candidate must at least roughly match its mesh
MIN_CONF = 0.3       # GroundingDINO confidence needed to (re-)prompt


def pack(m):
    return np.packbits(m.ravel()), m.shape


def unpack(p):
    bits, shape = p
    return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def box_mask_centroid(m):
    v, u = np.nonzero(m)
    return np.array([u.mean(), v.mean()]) if len(u) else None


class Driver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run = Path(cfg["run_dir"])
        self.objs = cfg["objects"]  # [{id, name, mesh_ext}]
        self.oid = [o["id"] for o in self.objs]
        self.n = cfg["n_frames"]
        ki = json.load(open(self.run / "recon_hawor/intrinsics_stable.json"))
        self.K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
        self.work = Path(cfg["work_dir"])
        self.log = {"passes": [], "refs": {}, "prompts": {}}
        self._depth = {}

    @property
    def by_id(self):
        return {o["id"]: o for o in self.objs}

    @property
    def by_name(self):
        return {o["name"]: o for o in self.objs}

    # ---------- IO ----------
    def depth(self, t):
        if t not in self._depth:
            if len(self._depth) > 64:
                self._depth.pop(next(iter(self._depth)))
            self._depth[t] = decode_depth(self.run / f"depth_stereo/{t:06d}.png")
        return self._depth[t]

    def rgb(self, t):
        return np.asarray(Image.open(self.run / f"recon_hawor/frames/{t:06d}.png").convert("RGB"))

    # ---------- models ----------
    def load_models(self):
        self.seg = self.cfg.get("segmenter", "sam2_gdino")
        if self.seg == "sam3":
            from seg_sam3 import Sam3Segmenter
            self.sam3 = Sam3Segmenter(self.cfg["sam3_weights_dir"], self.run / "recon_hawor/frames", self.n)
            return
        from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        wd = self.cfg["weights_dir"]
        os.environ["HYDRA_CONFIG_SEARCH_PATH"] = wd
        self.vp = build_sam2_video_predictor_low_mem(config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                                                     ckpt_path=os.path.join(wd, "sam2.1_hiera_large.pt"))
        self.ip = SAM2ImagePredictor(self.vp)

    # ---------- 1) candidates ----------
    def build_candidates(self, dets):
        """cands[t] = list of {mask(packed), box, conf{name: c}, geo{name: rec}}"""
        cands = {}
        names = [o["name"] for o in self.objs]
        if self.seg == "sam3":  # text-prompted instance masks on the sampled frames
            dets = {str(t): None for t in range(0, self.n, self.cfg.get("stride", 5))}
        for ts, dl in dets.items():
            t = int(ts)
            rgb = self.rgb(t)
            if self.seg == "sam3":
                out = [({"object": nm, "confidence": c, "box": b}, m, c)
                       for nm, c, b, m in self.sam3.detect(rgb, self.cfg["text_prompts"])]
            else:
                dl = [d for d in dl if d["confidence"] >= 0.25]
                if not dl:
                    continue
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    self.ip.set_image(rgb)
                    out = []
                    for d in dl:
                        b = d["box"]
                        m, s, _ = self.ip.predict(box=np.array([b["x0"], b["y0"], b["x1"], b["y1"]]),
                                                  multimask_output=False)
                        out.append((d, m[0] > 0, float(s[0])))
            merged = []
            for d, m, s in out:
                if m.sum() < 150:
                    continue
                for c in merged:
                    if mask_iou(c["_m"], m) > MERGE_IOU:
                        c["conf"][d["object"]] = max(c["conf"].get(d["object"], 0), d["confidence"])
                        if d["confidence"] > c["best"]:
                            c["best"], c["box"] = d["confidence"], d["box"]
                        break
                else:
                    merged.append({"_m": m, "box": d["box"], "best": d["confidence"],
                                   "conf": {d["object"]: d["confidence"]}, "sam": s})
            dep = self.depth(t)
            for c in merged:
                c["geo"] = {o["name"]: geometry_check(c["_m"], dep, self.K, o["mesh_ext"]) for o in self.objs}
                c["mask"] = pack(c.pop("_m"))
            cands[t] = merged
        self.cands = cands
        return cands

    def score(self, c, o):
        g = c["geo"][o["name"]]
        if not g["ok"] or g["reason"] == "no_depth":
            return 0.0
        own = c["conf"].get(o["name"], 0.0)
        if own == 0.0:
            # The candidate was found by some other object's prompt. That is worth
            # something for look-alikes the detector ranks wrongly (pot vs. lid), but
            # only if geometry does not clearly prefer whoever did claim it -
            # otherwise the water pitcher adopts the blue cup's mask (ep21/ep22).
            others = {k: v for k, v in c["conf"].items() if k != o["name"]}
            if not others:
                return 0.0
            claimer = max(others, key=others.get)
            if claimer in self.by_name and c["geo"][claimer]["fit"] > g["fit"] + 0.05:
                return 0.0
            own = 0.5 * others[claimer]
        return own * (0.3 + 0.7 * g["fit"])

    def assign(self, t, extra=None):
        """Hungarian object<->candidate assignment at sampled frame t.
        extra(o, c) -> multiplicative weight (e.g. temporal continuity)."""
        cl = self.cands.get(t, [])
        if not cl:
            return {}
        S = np.zeros((len(self.objs), len(cl)))
        for i, o in enumerate(self.objs):
            for j, c in enumerate(cl):
                S[i, j] = self.score(c, o) * (extra(o, c) if extra else 1.0)
        r, cidx = linear_sum_assignment(-S)
        return {self.objs[i]["id"]: (cl[j], S[i, j]) for i, j in zip(r, cidx) if S[i, j] > 0}

    def ref_ok(self, c, o):
        """A candidate may only become an object's REFERENCE if its mesh-size fit is
        decent and no other object in the episode explains it clearly better. Without
        this the pitcher took the blue cup's mask as its reference and then tracked
        the cup for the whole episode (ep21/ep22)."""
        g = c["geo"][o["name"]]
        if not g["ok"] or g["fit"] < REF_MIN_FIT:
            return False
        return all(c["geo"][o2["name"]]["fit"] <= g["fit"] + DUP_FIT_MARGIN
                   for o2 in self.objs if o2["id"] != o["id"])

    def choose_refs(self):
        best, best_key = None, None
        for t in sorted(self.cands):
            asg = {k: v for k, v in self.assign(t).items()
                   if self.ref_ok(v[0], self.by_id[k])}
            key = (len(asg), sum(s for _, s in asg.values()))
            if best_key is None or key > best_key:
                best, best_key = t, key
        refs = {}
        if best is not None:
            asg = {k: v for k, v in self.assign(best).items()
                   if self.ref_ok(v[0], self.by_id[k])}
            for oid, (c, s) in asg.items():
                refs[oid] = {"frame": best, "cand": c, "score": s}
        for o in self.objs:
            if o["id"] in refs:
                continue
            cand = []
            for t, cl in self.cands.items():
                for c in cl:
                    s = self.score(c, o)
                    if s <= 0 or not self.ref_ok(c, o):
                        continue
                    m = unpack(c["mask"])
                    clash = any(r["frame"] == t and mask_iou(unpack(r["cand"]["mask"]), m) > DUP_IOU
                                for r in refs.values())
                    if not clash:
                        cand.append((s, t, c))
            if cand:
                s, t, c = max(cand, key=lambda x: x[0])
                refs[o["id"]] = {"frame": t, "cand": c, "score": s}
        return refs

    # ---------- 2/3) video propagation ----------
    def propagate(self, prompts, hand_prompts):
        """prompts: {oid: [(frame, packed mask)]}. Returns {oid: {t: packed}}.
        Backward pass from the LAST prompt frame, forward pass from the first; per
        object, frames before its own first prompt come from the backward pass
        (memory chained from its prompt), later frames from the forward pass."""
        if self.seg == "sam3":
            first = {oid: min(t for t, _ in pl) for oid, pl in prompts.items()}
            hp = [(h["frame_index"], h["object_id"], np.asarray(Image.open(h["mask_path"]).convert("L")) > 0)
                  for h in hand_prompts]
            for t, oid, _ in hp:
                first[oid] = min(first.get(oid, 10 ** 9), t)
            last = max([t for pl in prompts.values() for t, _ in pl] + [t for t, _, _ in hp])
            out = self.sam3.propagate({oid: [(t, unpack(pm)) for t, pm in pl] for oid, pl in prompts.items()},
                                      hp, first, last)
            return {oid: {t: pack(m) for t, m in d.items()} for oid, d in out.items()}
        st = self.state
        self.vp.reset_state(st)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for oid, pl in prompts.items():
                for t, pm in pl:
                    self.vp.add_new_mask(st, frame_idx=t, obj_id=oid, mask=unpack(pm))
            for hp in hand_prompts:
                m = np.asarray(Image.open(hp["mask_path"]).convert("L")) > 0
                self.vp.add_new_mask(st, frame_idx=hp["frame_index"], obj_id=hp["object_id"], mask=m)
            first = {oid: min(t for t, _ in pl) for oid, pl in prompts.items()}
            for hp in hand_prompts:
                first[hp["object_id"]] = min(first.get(hp["object_id"], 10 ** 9), hp["frame_index"])
            last = max(first.values()) if first else 0
            last = max([last] + [t for pl in prompts.values() for t, _ in pl])
            res = {}
            for reverse in (True, False):
                kw = {"start_frame_idx": last} if reverse else {}
                for t, ids, masks in self.vp.propagate_in_video(st, reverse=reverse, **kw):
                    for i, oid in enumerate(ids):
                        if reverse or t >= first.get(oid, 0) or t not in res.get(oid, {}):
                            res.setdefault(oid, {})[t] = pack((masks[i, 0] > 0).cpu().numpy())
        return res

    def nested(self, ida, ma, idb, mb):
        """Legitimate overlap: a smaller object sitting on/in a larger one (lid on
        pot) - the larger object's mask strictly contains the smaller one's."""
        if any(o["mesh_ext"] is None for o in self.objs):   # size-free mode: the larger MASK is the container
            (sm, bm) = sorted([ma, mb], key=lambda m: m.sum())
            return bm.sum() >= 1.3 * sm.sum() and containment(sm, bm) >= 0.85
        diag = {o["id"]: float(np.linalg.norm(o["mesh_ext"])) for o in self.objs}
        (sid, sm), (bid, bm) = sorted([(ida, ma), (idb, mb)], key=lambda x: diag[x[0]])
        return (diag[bid] > 1.2 * diag[sid] and bm.sum() >= 1.08 * sm.sum()
                and containment(sm, bm) >= 0.85)

    def validate(self, res):
        """flags[oid][t] in {'ok','empty','too_large','too_small','dup','swap'}"""
        flags, geos = {o["id"]: {} for o in self.objs}, {o["id"]: {} for o in self.objs}
        for t in range(self.n):
            dep = self.depth(t)
            ms = {o["id"]: unpack(res[o["id"]][t]) if t in res.get(o["id"], {}) else None for o in self.objs}
            for o in self.objs:
                m = ms[o["id"]]
                if m is None or m.sum() < 150:
                    flags[o["id"]][t] = "empty"
                    geos[o["id"]][t] = None
                    continue
                g = geometry_check(m, dep, self.K, o["mesh_ext"])
                geos[o["id"]][t] = g
                flags[o["id"]][t] = "ok" if g["ok"] else g["reason"]
            # duplicates: the id whose geometry fits worse is the wrong one
            ids = [o["id"] for o in self.objs]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    ma, mb = ms[ids[a]], ms[ids[b]]
                    if ma is None or mb is None or not ma.any() or not mb.any():
                        continue
                    if mask_iou(ma, mb) > DUP_IOU and not self.nested(ids[a], ma, ids[b], mb):
                        # Two ids on one mask: the mask belongs to whichever object's
                        # mesh it actually fits (the pitcher and the cup differ 3x in
                        # size), so keep that one and only flag the other. Both are
                        # flagged only when the sizes genuinely cannot separate them.
                        fa = (geos[ids[a]][t] or {}).get("fit", 0)
                        fb = (geos[ids[b]][t] or {}).get("fit", 0)
                        if abs(fa - fb) > DUP_FIT_MARGIN:
                            loser = ids[a] if fa < fb else ids[b]
                            flags[loser][t] = "dup"
                        else:
                            for oid in (ids[a], ids[b]):
                                flags[oid][t] = "dup"
            # identity disagreement at sampled frames with a confident assignment
            if t in self.cands:
                asg = self.assign(t, self.continuity(res, flags, t))
                for o in self.objs:
                    m = ms[o["id"]]
                    if o["id"] not in asg or m is None or flags[o["id"]][t] != "ok":
                        continue
                    mine = unpack(asg[o["id"]][0]["mask"])
                    for o2 in self.objs:
                        if o2["id"] == o["id"] or o2["id"] not in asg:
                            continue
                        other = unpack(asg[o2["id"]][0]["mask"])
                        if mask_iou(m, other) > 0.5 and mask_iou(m, mine) < 0.2 and asg[o2["id"]][1] >= MIN_CONF:
                            flags[o["id"]][t] = "swap"
        return flags, geos

    def continuity(self, res, flags, t, window=30):
        """Weight for candidate c of object o at frame t: closeness of c to the
        object's mask at the nearest frame where its track was valid."""
        ref = {}
        for o in self.objs:
            fl = flags.get(o["id"], {})
            for dt in range(0, window + 1):
                hit = None
                for tt in (t - dt, t + dt):
                    if 0 <= tt < self.n and fl.get(tt) == "ok" and tt in res.get(o["id"], {}):
                        hit = tt
                        break
                if hit is not None:
                    ref[o["id"]] = (box_mask_centroid(unpack(res[o["id"]][hit])), abs(hit - t))
                    break
        diag = float(np.hypot(*self.K[:2, 2] * 2))

        def w(o, c):
            if o["id"] not in ref or ref[o["id"]][0] is None:
                return 1.0
            cc = box_mask_centroid(unpack(c["mask"]))
            d = np.linalg.norm(cc - ref[o["id"]][0]) / diag
            return float(np.exp(-d / (0.08 + 0.01 * ref[o["id"]][1])))
        return w

    def reprompt(self, res, flags, prompts, max_per_obj=10, min_gap=20):
        added = 0
        for o in self.objs:
            oid = o["id"]
            bad = [t for t in range(self.n) if flags[oid][t] in ("too_large", "dup", "swap", "empty", "too_small")]
            if not bad:
                continue
            have = [t for t, _ in prompts[oid]]
            if len(have) >= max_per_obj:
                continue
            # candidate frames: sampled frames that are bad for this object
            for t in sorted(set(bad) & set(self.cands)):
                if len(prompts[oid]) >= max_per_obj:
                    break
                if any(abs(t - h) < min_gap for h in have):
                    continue
                asg = self.assign(t, self.continuity(res, flags, t))
                if oid not in asg or asg[oid][1] < MIN_CONF * 0.5:
                    continue
                c = asg[oid][0]
                if c["conf"].get(o["name"], 0) < MIN_CONF and asg[oid][1] < MIN_CONF:
                    continue
                cm = unpack(c["mask"])
                # never prompt onto a mask that another id holds validly
                clash = False
                for o2 in self.objs:
                    if o2["id"] != oid and t in res.get(o2["id"], {}) and flags[o2["id"]][t] == "ok":
                        if mask_iou(cm, unpack(res[o2["id"]][t])) > DUP_IOU:
                            clash = True
                if clash:
                    continue
                prompts[oid].append((t, c["mask"]))
                have.append(t)
                self.log["prompts"].setdefault(o["name"], []).append(
                    {"frame": t, "reason": flags[oid][t], "score": round(asg[oid][1], 3)})
                added += 1
        return added

    # ---------- main ----------
    def run_all(self):
        t0 = time.time()
        self.load_models()
        dets = json.load(open(self.cfg["dets"])) if self.seg != "sam3" else None
        self.build_candidates(dets)
        print(f"candidates built ({sum(len(v) for v in self.cands.values())}) {time.time()-t0:.0f}s", flush=True)
        refs = self.choose_refs()
        for o in self.objs:
            r = refs.get(o["id"])
            self.log["refs"][o["name"]] = None if r is None else {
                "frame": r["frame"], "score": round(r["score"], 3), "box": r["cand"]["box"],
                "conf": r["cand"]["conf"], "geo": r["cand"]["geo"][o["name"]]}
            print(f"ref {o['name']}: {self.log['refs'][o['name']] and {k: v for k, v in self.log['refs'][o['name']].items() if k != 'geo'}}", flush=True)
        self.objs = [o for o in self.objs if o["id"] in refs]
        if not self.objs:
            raise SystemExit("no object could be detected")
        hand_prompts = self.cfg["hand_prompts"]
        if self.seg != "sam3":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.state = self.vp.init_state(self.cfg["video"])
        prompts = {o["id"]: [(refs[o["id"]]["frame"], refs[o["id"]]["cand"]["mask"])] for o in self.objs}
        for p in range(self.cfg.get("passes", 3)):
            res = self.propagate(prompts, hand_prompts)
            flags, geos = self.validate(res)
            summ = {o["name"]: {k: sum(1 for v in flags[o["id"]].values() if v == k)
                                for k in ("ok", "empty", "too_large", "too_small", "dup", "swap", "no_depth")}
                    for o in self.objs}
            self.log["passes"].append(summ)
            print(f"pass {p}: {json.dumps(summ)} {time.time()-t0:.0f}s", flush=True)
            if p == self.cfg.get("passes", 3) - 1 or self.reprompt(res, flags, prompts) == 0:
                break
        self.finalize(res, flags, geos)
        self.log["ref_frames"] = {o["name"]: refs[o["id"]]["frame"] for o in self.objs}
        self.log["ref_boxes"] = {o["name"]: refs[o["id"]]["cand"]["box"] for o in self.objs}
        self.log["ref_conf"] = {o["name"]: refs[o["id"]]["cand"]["conf"].get(o["name"], 0) for o in self.objs}
        json.dump(self.log, open(self.cfg["report"], "w"), indent=1, default=float)
        print(f"done {time.time()-t0:.0f}s", flush=True)

    def finalize(self, res, flags, geos):
        out = Path(self.cfg["masks_dir"])
        oute = Path(self.cfg["masks_ekf_dir"])
        final = {o["name"]: {"kept": 0, "cleaned": 0, "dropped": {}} for o in self.objs}
        for hid in HAND_IDS:
            if hid in res:
                (out / str(hid)).mkdir(parents=True, exist_ok=True)
                for t, pm in res[hid].items():
                    Image.fromarray(unpack(pm).astype(np.uint8) * 255).save(out / str(hid) / f"{t:06d}.png")
        for o in self.objs:
            (out / str(o["id"])).mkdir(parents=True, exist_ok=True)
            (oute / str(o["id"])).mkdir(parents=True, exist_ok=True)
        for t in range(self.n):
            dep = self.depth(t)
            keep = {}
            for o in self.objs:
                oid, f = o["id"], flags[o["id"]][t]
                m = unpack(res[oid][t]) if t in res.get(oid, {}) else None
                if m is not None and f in ("too_large", "ok"):
                    m2 = clean_mask(m, dep, self.K, o["mesh_ext"])
                    if f == "too_large":
                        g = geometry_check(m2, dep, self.K, o["mesh_ext"])
                        if g["ok"]:
                            f = "ok"
                            final[o["name"]]["cleaned"] += 1
                    m = m2
                if f == "ok" and m is not None and m.sum() >= 150:
                    keep[oid] = m
                else:
                    final[o["name"]]["dropped"][f] = final[o["name"]]["dropped"].get(f, 0) + 1
            for o in self.objs:
                oid = o["id"]
                m = keep.get(oid)
                em = m if m is not None else np.zeros(dep.shape, bool)
                Image.fromarray(em.astype(np.uint8) * 255).save(oute / str(oid) / f"{t:06d}.png")
                if m is not None:
                    Image.fromarray(m.astype(np.uint8) * 255).save(out / str(oid) / f"{t:06d}.png")
                    final[o["name"]]["kept"] += 1
        self.log["final"] = final
        print("final:", json.dumps(final), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    a = ap.parse_args()
    Driver(json.load(open(a.cfg))).run_all()


if __name__ == "__main__":
    main()
