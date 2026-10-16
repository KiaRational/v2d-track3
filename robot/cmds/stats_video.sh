python - <<'PY'
import pyarrow.parquet as pq, numpy as np, glob
f = glob.glob("/scratch/kghasemz/v2d/robot/hmd/ego_recon/processed/sequence_id=ep39_dustpan/robot_name=sharpa_wave/*.parquet")[0]
t = pq.read_table(f).to_pylist()[0]
A = lambda k: np.asarray(t[k], float)
op = A("object_root_position"); print("object z min/max", op[:, 2].min().round(3), op[:, 2].max().round(3), " xy travel", np.ptp(op[:, :2], 0).round(3))
for s in ("right", "left"):
    c = A(f"mano_{s}_object_contact_positions"); nz = (np.abs(c).sum(-1) > 0)
    w = A(f"robot_{s}_wrist_position"); d = np.linalg.norm(w - op, axis=1)
    print(s, "frames with any contact", nz.any(1).sum(), "/", len(nz), " contacts/frame max", nz.sum(1).max(),
          " wrist z min/med", w[:, 2].min().round(3), np.median(w[:, 2]).round(3), " wrist-object dist min", d.min().round(3),
          " retarget task err med", np.median(A(f"robot_{s}_frame_task_errors")).round(3))
lift = np.where(op[:, 2] > op[:, 2].min() + 0.02)[0]
print("lift frames (>2cm above rest):", (lift.min(), lift.max(), len(lift)) if len(lift) else None)
PY
# MP4 rollouts (unassisted, deterministic) of chosen ep39 checkpoints. Args via env CKPTS="10000384 90003456".
O=/scratch/kghasemz/v2d/robot/fc_out/ep39_dustpan; mkdir -p $O/video
# markers.enabled=false "viewer.camera_position=[0.55,-0.75,0.45]" "viewer.camera_target=[0.0,-0.12,0.05]": the marker overlay crashes the GL viewer at this revision (scalar point width).
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true  # no X on compute nodes: pyglet EGL headless
for s in $CKPTS; do
  python scripts/view_policy.py evaluation.checkpoint=$O/policy_$s.safetensors viewer.backend=mp4 viewer.headless=true \
    viewer.realtime=false viewer.output_path=$O/video/policy_$s.mp4 viewer.num_frames=440 viewer.video_fps=30 \
    viewer.width=1280 viewer.height=720 markers.enabled=false "viewer.camera_position=[0.55,-0.75,0.45]" "viewer.camera_target=[0.0,-0.12,0.05]" 2>&1 | grep -v "^Module\|nefc" | tail -12
done
ls -la $O/video
