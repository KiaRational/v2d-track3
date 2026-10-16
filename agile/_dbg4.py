import sys, json, numpy as np, cv2, torch
sys.path.insert(0, "/code/agile")
import run_agile as A
from render import Renderer, sample_surface
from dino import Dino
from scipy.spatial.transform import Rotation
ep = A.ROOT / "runs/public_ep000011"; t = 24; dev = "cuda"
intr = json.load(open(ep / "recon_hawor/intrinsics_stable.json")); K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1.]])
obj = json.load(open(ep / "agile/beige_cup/prepare.json"))
fp = json.load(open(obj["fp_pose_at_iof"])); q = fp["rotation"]
R0 = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T0 = np.array(fp["translation"])
rend = Renderer(obj["mesh"], dev); surf = torch.tensor(sample_surface(obj["mesh"]), device=dev)
dino = Dino(str(A.ROOT / "weights/dinov3/dinov3-vitl16-pretrain-lvd1689m"), dev)
scale = torch.ones(3, device=dev)
of = A.ObjectFrame(ep, t, obj, K, (1520, 2028), R0, T0, scale, rend, dino, [2, 3, 1], dev)
rgb, z = rend.textured(torch.tensor(R0, dtype=torch.float32, device=dev), torch.tensor(T0, dtype=torch.float32, device=dev), scale, of.K, A.CROP_RES)
cv2.imwrite("/scratch/kghasemz/v2d/cache/dbg4_textured.png", (rgb.cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
print("textured render mean rgb:", rgb.mean((0, 1)).cpu().numpy().round(3), "covered px:", int(torch.isfinite(z).sum()))
F_saved = of.F_gt
for wm, wd in ((5.0, 0.0), (0.0, 10.0), (5.0, 10.0)):
    of.F_gt = F_saved if wd else None
    A.W_MASK, A.W_DINO = wm, wd
    R, T, log = A.optimise_object(of, rend, dino if wd else None, surf, R0, T0, scale, None, None, None, None, 400)
    print(f"w_mask={wm} w_dino={wd}: T {T.round(3)} (FP {T0.round(3)})  log {({k: round(v, 3) for k, v in log.items()})}")
