# MP4 rollouts (unassisted, deterministic) of chosen ep39 checkpoints. Args via env CKPTS="10000384 90003456".
O=/scratch/kghasemz/v2d/robot/fc_out/ep39_dustpan; mkdir -p $O/video
# markers.enabled=false: the marker overlay crashes the GL viewer at this revision (scalar point width).
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true  # no X on compute nodes: pyglet EGL headless
for s in $CKPTS; do
  python scripts/view_policy.py evaluation.checkpoint=$O/policy_$s.safetensors viewer.backend=mp4 viewer.headless=true \
    viewer.realtime=false viewer.output_path=$O/video/policy_$s.mp4 viewer.num_frames=440 viewer.video_fps=30 \
    viewer.width=1280 viewer.height=720 markers.enabled=false "viewer.camera_position=[0.55,-0.75,0.45]" "viewer.camera_target=[0.0,-0.12,0.05]" 2>&1 | grep -v "^Module\|nefc" | tail -12
done
ls -la $O/video
