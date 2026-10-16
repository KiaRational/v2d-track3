import sys, json, numpy as np, torch
sys.path.insert(0, "/code/agile")
import run_agile as A
from render import Renderer
from scipy.spatial.transform import Rotation
ep = A.ROOT / "runs/public_ep000011"; t = 24; dev = "cuda"
intr = json.load(open(ep / "recon_hawor/intrinsics_stable.json")); K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1.]])
obj = json.load(open(ep / "agile/beige_cup/prepare.json")); fp = json.load(open(obj["fp_pose_at_iof"])); q = fp["rotation"]
R0 = torch.tensor(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), dtype=torch.float32, device=dev)
T0 = torch.tensor(fp["translation"], dtype=torch.float32, device=dev)
rend = Renderer(obj["mesh"], dev); s = torch.ones(3, device=dev)
of = A.ObjectFrame(ep, t, obj, K, (1520, 2028), R0.cpu().numpy(), T0.cpu().numpy(), s, rend, None, [2, 3, 1], dev)
f = lambda T: A.mask_loss(rend.silhouette(R0, T, s, of.K, A.CROP_RES), of.mask_gt)
T = T0.clone().requires_grad_(True); L = f(T); L.backward()
print("loss at FP pose:", round(float(L), 4), " autograd dL/dT:", T.grad.cpu().numpy())
for i, e in enumerate(np.eye(3) * 0.004):
    e = torch.tensor(e, dtype=torch.float32, device=dev)
    print(f"  finite diff dL/dT[{i}]:", round(float((f(T0 + e) - f(T0 - e)) / 0.008), 3))
T = T0.clone().requires_grad_(True); opt = torch.optim.Adam([T], lr=1e-3); traj = []
for it in range(150):
    L = f(T); opt.zero_grad(); L.backward(); opt.step(); traj.append(float(L))
print("Adam mask-only loss:", [round(x, 3) for x in traj[:5]], "...", [round(x, 3) for x in traj[-3:]], " T", T.detach().cpu().numpy().round(3))
