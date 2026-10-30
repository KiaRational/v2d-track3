"""CPU before/after video and close-up contact sheets for evidence_fusion (Slurm only)."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

from object_icp import load_T
from evidence_fusion import geometry_score, scan_frame, metrics, angle


def overlay(frame, mesh, T, K):
    out = frame.copy()
    P = mesh.vertices @ T[:3,:3].T + T[:3,3]
    uv = np.stack([K["fx"]*P[:,0]/P[:,2]+K["cx"],K["fy"]*P[:,1]/P[:,2]+K["cy"]],1)
    good = np.isfinite(uv).all(1) & (P[:,2]>.01)
    for a,b in mesh.edges_unique[::max(1,len(mesh.edges_unique)//2000)]:
        if good[a] and good[b] and np.abs(uv[[a,b]]).max()<100000:
            cv2.line(out,tuple(uv[a].astype(int)),tuple(uv[b].astype(int)),(40,230,60),1,cv2.LINE_AA)
    return out,uv[good]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs",nargs="+")
    ap.add_argument("--variant",default="fused_evidence_recon")
    ap.add_argument("--evidence",default="fused_evidence_recon")
    a=ap.parse_args()
    root=Path(os.environ["V2D_ROOT"])
    for rn in a.runs:
        run=root/"runs"/rn
        report=json.loads((run/a.evidence/"evidence_report.json").read_text())
        K=json.loads((run/"recon_hawor/intrinsics_stable.json").read_text())
        cap=cv2.VideoCapture(str(run/"input_undistorted.mp4"))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dest=root/"viz"/a.variant/rn
        dest.mkdir(parents=True,exist_ok=True)
        audit={"variant":a.variant,"objects":{}}
        objects=json.loads((run/"objects_recon/objects.json").read_text())
        plane=json.loads((run/"table_plane.json").read_text()) if (run/"table_plane.json").exists() else None
        for name,detail in report["objects"].items():
            mesh=trimesh.load(root/"meshes_recon"/name/f"{name}.obj",force="mesh")
            rows=[]; before_scores=[]; after_scores=[]
            pts,fi=trimesh.sample.sample_surface(mesh,2400,seed=0)
            obj=next(o for o in objects if o["name"]==name)
            other=[2,3]+[o["id"] for o in objects if o["id"]!=obj["id"]]
            for t in range(n):
                pose=run/a.variant/name/"poses"/f"{t:06d}.json"
                if not pose.exists(): break
                S=load_T(run/"recon_hawor/slam_poses"/pose.name)
                rows.append(S@load_T(pose))
                if str(t) in detail["evidence"]:
                    scan=scan_frame(run,obj["id"],other,t,K)
                    if scan is not None:
                        before_scores.append(geometry_score(load_T(run/report["args"]["src"]/name/"poses"/pose.name),pts,mesh.face_normals[fi],scan))
                        after_scores.append(geometry_score(load_T(pose),pts,mesh.face_normals[fi],scan))
            W=np.array(rows)
            stats=metrics(W,np.asarray(mesh.centroid),report["fps"])
            stats.update(geometry_before_mm=float(np.mean(before_scores)*1000),geometry_after_mm=float(np.mean(after_scores)*1000))
            if plane:
                from motion_quality import table_slab_depth
                import torch
                vw=pts@np.swapaxes(W[:,:3,:3],1,2)+W[:,:3,3][:,None]
                depth=table_slab_depth(torch.tensor(vw),plane,side="auto").amax(1).numpy()*1000
                stats["table_penetration_p95_mm"]=float(np.percentile(depth,95))
                stats["table_penetration_max_mm"]=float(depth.max())
            stats["geometry_pass"]=bool(stats["geometry_after_mm"]<=stats["geometry_before_mm"]+1)
            audit["objects"][name]=stats
            print(rn,name,json.dumps(stats),flush=True)
            times=set(np.linspace(0,n-1,5).astype(int))
            times.update(sorted((int(t) for t in detail["evidence"]),
                key=lambda t:detail["evidence"][str(t)]["rotation_deg"],reverse=True)[:3])
            strips=[]
            for t in sorted(times):
                before=run/report["args"]["src"]/name/"poses"/f"{t:06d}.json"
                after=run/a.variant/name/"poses"/f"{t:06d}.json"
                if not before.exists() or not after.exists():
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES,t)
                ok,frame=cap.read()
                if not ok: continue
                b,u=overlay(frame,mesh,load_T(before),K)
                c,v=overlay(frame,mesh,load_T(after),K)
                xy=np.concatenate([u,v])
                x0,y0=np.maximum(np.floor(xy.min(0)-60).astype(int),0)
                x1,y1=np.minimum(np.ceil(xy.max(0)+60).astype(int),[frame.shape[1],frame.shape[0]])
                if x1<=x0 or y1<=y0: continue
                panels=[]
                for im,label in ((frame,"Image"),(b,"Before"),(c,"Refined")):
                    crop=im[y0:y1,x0:x1]
                    scale=min(380/crop.shape[1],300/crop.shape[0])
                    crop=cv2.resize(crop,None,fx=scale,fy=scale)
                    tile=np.zeros((330,400,3),np.uint8)
                    tile[30:30+crop.shape[0],:crop.shape[1]]=crop
                    cv2.putText(tile,f"{label}  frame {t}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                    panels.append(tile)
                strips.append(np.concatenate(panels,axis=1))
            if strips:
                path=dest/f"{name}.jpg"
                cv2.imwrite(str(path),np.concatenate(strips,axis=0))
                print(path,flush=True)
        cap.release()
        (dest/"validation.json").write_text(json.dumps(audit,indent=2))


if __name__=="__main__": main()
