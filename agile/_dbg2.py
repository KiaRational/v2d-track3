import sys, json, numpy as np, cv2, torch
sys.path.insert(0, "/code/agile")
from render import Renderer, Crop
R = "/scratch/kghasemz/v2d/runs/public_ep000011"; t = 24; dev = "cuda"
intr = json.load(open(f"{R}/recon_hawor/intrinsics_stable.json")); K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1.]])
rend = Renderer("/scratch/kghasemz/v2d/meshes_obj/beige_cup/beige_cup.obj", dev)
p = json.load(open(f"{R}/smoke_hc2/beige_cup/poses/{t:06d}.json")); Rm, T = np.array(p["R"]), np.array(p["T"])
mask = cv2.imread(f"{R}/objects/masks/4/{t:06d}.png", 0) > 0
ys, xs = np.nonzero(mask); crop = Crop.around([xs.min(), ys.min(), xs.max(), ys.max()], (2028, 1520), 256, expand=2.5)
Kc = crop.K(torch.tensor(K, dtype=torch.float32, device=dev))
for name, TT in (("agile_T", T), ("toolkit_fp_T", np.array(json.load(open(f"{R}/objects/beige_cup/poses_smoothed/{t:06d}.json"))["translation"]))):
    sil = rend.silhouette(torch.tensor(Rm, dtype=torch.float32, device=dev), torch.tensor(TT, dtype=torch.float32, device=dev), torch.ones(3, device=dev), Kc, 256).detach().cpu().numpy()
    img = cv2.imread(f"{R}/recon_hawor/frames/{t:06d}.png"); ci = crop.image(img, cv2.INTER_LINEAR); cm = crop.image(mask.astype(np.float32), cv2.INTER_NEAREST)
    vis = ci.copy(); vis[cm > 0.5] = (0.5 * vis[cm > 0.5] + [0, 127, 0]).astype(np.uint8); vis[sil > 0.5] = (0.5 * vis[sil > 0.5] + [0, 0, 127]).astype(np.uint8)
    cv2.imwrite(f"/scratch/kghasemz/v2d/cache/dbg_{name}.png", vis); print(name, TT.round(3), "sil px", int((sil > 0.5).sum()), "mask px", int((cm > 0.5).sum()))
