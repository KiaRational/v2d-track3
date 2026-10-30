"""Exact hand-relative pose during confirmed carries; no soft rotational slip.

Uses evidence_fusion carry seeds, image-space release evidence and the source
pose immediately before pickup. Writes a new variant, never overwrites inputs.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from evidence_fusion import angle, robust_pose, metrics
from fuse_objects import stereo_hands
from object_icp import load_T


def lock_interval(W, H, start, stop, anchor_frames=5):
    """Preserve the pre-pickup world orientation, then keep one exact palm offset."""
    out=W.copy()
    # At pickup, orientation from the preceding visible resting interval is more
    # reliable than a majority vote across already-corrupted held-frame poses.
    lo=max(0,start-anchor_frames)
    ref=robust_pose(W[lo:start+1],np.ones(start-lo+1))
    ref[:3,3]=W[start,:3,3]
    G=np.linalg.inv(H[start])@ref
    out[start:stop]=H[start:stop]@G
    return out,G


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--source",default="fused_bfix_recon")
    ap.add_argument("--evidence",default="fused_evidence_recon")
    ap.add_argument("--out",default="fused_rigid_verified_recon")
    ap.add_argument("--hands",default="hands_c")
    ap.add_argument("--objects",nargs="+",required=True)
    a=ap.parse_args()
    root=Path(os.environ["V2D_ROOT"]); run=root/"runs"/a.run; dest=run/a.out
    if dest.exists(): raise FileExistsError(dest)
    report=json.loads((run/a.evidence/"evidence_report.json").read_text())
    source_objects=json.loads((run/"objects_recon/objects.json").read_text())
    S=np.array([load_T(p) for p in sorted((run/"recon_hawor/slam_poses").glob("[0-9]*.json"))])
    n=len(S); hands=stereo_hands(run,n,a.hands)
    K=json.loads((run/"recon_hawor/intrinsics_stable.json").read_text())
    summary={"source":a.source,"hands":a.hands,"objects":{}}
    for obj in source_objects:
        name=obj["name"]
        W=S@np.array([load_T(run/a.source/name/"poses"/f"{t:06d}.json") for t in range(n)])
        out=W.copy(); spans=[]
        if name in a.objects:
            seeds=report["objects"][name]["segments"]
            occupied=np.zeros(n,bool)
            # Longest confirmed contact first; avoid conflicting hard bimanual locks.
            for seed in sorted(seeds,key=lambda s:s["to"]-s["from"],reverse=True):
                side=seed["side"]; hd=hands[side]; H=S@hd["H"]
                valid=np.isfinite(H).all((1,2)); near=np.zeros(n,bool)
                for t in np.flatnonzero(valid):
                    p=run/"objects_recon/masks"/str(obj["id"])/f"{t:06d}.png"
                    if not p.exists(): continue
                    m=cv2.imread(str(p),0)
                    m=cv2.resize(m,None,fx=.25,fy=.25,interpolation=cv2.INTER_NEAREST)
                    m=cv2.dilate(m,np.ones((31,31),np.uint8))>0
                    J=hd["J"][t]; positive=np.isfinite(J).all(1)&(J[:,2]>.05)
                    J=J[positive]
                    uv=np.rint(np.stack([K["fx"]*J[:,0]/J[:,2]+K["cx"],K["fy"]*J[:,1]/J[:,2]+K["cy"]],1)*.25).astype(int)
                    inside=(uv[:,0]>=0)&(uv[:,0]<m.shape[1])&(uv[:,1]>=0)&(uv[:,1]<m.shape[0])
                    near[t]=bool(np.any(m[uv[inside,1],uv[inside,0]]))
                    # A drifting estimated palm must not invent a release while
                    # the segmented hand still touches the object in the image.
                    hp=p.parent.parent/str(2 if side=="left" else 3)/p.name
                    if hp.exists():
                        hm=cv2.imread(str(hp),0)
                        hm=cv2.resize(hm,(m.shape[1],m.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                        near[t] |= np.count_nonzero(hm&m)>=3
                start=seed["from"]; end=start
                while end<n and valid[end] and not occupied[end]:
                    if end>=start+3 and not near[max(start,end-2):end+1].any(): break
                    end+=1
                if end-start<4: continue
                # Smooth only this palm interval, without crossing release or gaps.
                smooth=H.copy()
                for t in range(start,end):
                    ids=np.arange(max(start,t-4),min(end,t+5))
                    weights=np.exp(-.5*((ids-t)/1.5)**2)
                    smooth[t]=robust_pose(H[ids],weights)
                out,G=lock_interval(out,smooth,start,end)
                occupied[start:end]=True
                local=np.linalg.inv(smooth[start:end])@out[start:end]
                err=np.degrees(angle(local[:,:3,:3],G[:3,:3]))
                spans.append({"side":side,"from":start,"to":end-1,
                              "relative_rotation_max_deg":float(err.max()),
                              "relative_translation_max_mm":float(np.linalg.norm(local[:,:3,3]-G[:3,3],axis=1).max()*1000)})
            if not spans: raise ValueError(f"No confirmed rigid carry for {name}")
        d=dest/name/"poses"; d.mkdir(parents=True)
        for t,T in enumerate(np.linalg.inv(S)@out):
            (d/f"{t:06d}.json").write_text(json.dumps({"R":T[:3,:3].tolist(),"T":T[:3,3].tolist()}))
        summary["objects"][name]={"rigid_intervals":spans}
        print(a.run,name,json.dumps(summary["objects"][name]),flush=True)
    (dest/"rigid_report.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__": main()
