import sys, json, numpy as np, cv2, torch
sys.path.insert(0, "/code/agile")
from render import Renderer, Crop
from scipy.spatial.transform import Rotation
R = "/scratch/kghasemz/v2d/runs/public_ep000011"; t = 24; dev = "cuda"
intr = json.load(open(f"{R}/recon_hawor/intrinsics_stable.json")); K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1.]])
rend = Renderer("/scratch/kghasemz/v2d/meshes_obj/beige_cup/beige_cup.obj", dev)
fp = json.load(open(f"{R}/agile/beige_cup/fp_ref000024/000024.json")); q = fp["rotation"]
Rm = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T = np.array(fp["translation"])
mask = cv2.imread(f"{R}/objects/masks/4/{t:06d}.png", 0) > 0
ys, xs = np.nonzero(mask); crop = Crop.around([xs.min(), ys.min(), xs.max(), ys.max()], (2028, 1520), 256, expand=1.6)
Kc = crop.K(torch.tensor(K, dtype=torch.float32, device=dev))
sil = rend.silhouette(torch.tensor(Rm, dtype=torch.float32, device=dev), torch.tensor(T, dtype=torch.float32, device=dev), torch.ones(3, device=dev), Kc, 256).detach().cpu().numpy() > 0.5
cm = crop.image(mask.astype(np.float32), cv2.INTER_NEAREST) > 0.5
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
print("IoU as rendered:", round(iou(sil, cm), 3), " v-flip:", round(iou(sil[::-1], cm), 3), " h-flip:", round(iou(sil[:, ::-1], cm), 3), " both:", round(iou(sil[::-1, ::-1], cm), 3))
# projected mesh vertices (pure numpy pinhole) vs rendered silhouette centroid
V = rend.verts.cpu().numpy() @ Rm.T + T; uv = V[:, :2] / V[:, 2:3] @ K[:2, :2].T + K[:2, 2]; uvc = (uv - [crop.x0, crop.y0]) * crop.s
print("numpy-projected verts centroid (crop px):", uvc.mean(0).round(1), " rendered sil centroid:", np.array(np.nonzero(sil))[::-1].mean(1).round(1), " mask centroid:", np.array(np.nonzero(cm))[::-1].mean(1).round(1))
img = crop.image(cv2.imread(f"{R}/recon_hawor/frames/{t:06d}.png"), cv2.INTER_LINEAR); vis = img.copy()
vis[cm] = (0.5 * vis[cm] + [0, 127, 0]).astype(np.uint8); vis[sil] = (0.5 * vis[sil] + [0, 0, 127]).astype(np.uint8)
cv2.imwrite("/scratch/kghasemz/v2d/cache/dbg3.png", vis)
