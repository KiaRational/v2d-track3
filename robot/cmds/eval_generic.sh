# Unassisted evaluation of every 20M-step checkpoint (+ the last) and MP4s of the last policy. Env: NAME.
O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/eval $O/video
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true
CK=$(ls $O/policy_*.safetensors | sort -V); LAST=$(echo "$CK" | tail -1)
for c in $(echo "$CK" | awk 'NR%2==0') $LAST; do
  s=$(basename $c .safetensors | cut -d_ -f2); [ -f $O/eval/metrics_$s.json ] && continue
  python scripts/evaluate_policy.py evaluation.checkpoint=$c evaluation.world_count=512 \
    evaluation.metrics_output=$O/eval/metrics_$s.json 2>&1 | grep -E "report|Error" | tail -3
done
s=$(basename $LAST .safetensors | cut -d_ -f2)
python scripts/view_policy.py evaluation.checkpoint=$LAST viewer.backend=mp4 viewer.headless=true viewer.realtime=false \
  viewer.output_path=$O/video/policy_$s.mp4 viewer.num_frames=600 viewer.video_fps=30 viewer.width=1280 viewer.height=720 \
  markers.enabled=false "viewer.camera_position=[${CAMP:-0.55,-0.75,0.45}]" "viewer.camera_target=[${CAMT:-0.0,-0.12,0.05}]" 2>&1 | grep -E "Video|Error"
