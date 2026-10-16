# Unassisted evaluation of saved checkpoints (learning curve) + MP4 rollouts of an early and the final policy.
O=/scratch/kghasemz/v2d/robot/fc_out/ep39_dustpan; mkdir -p $O/eval $O/video
CK=$(ls $O/policy_*.safetensors | sort -t_ -k2 -n)
N=$(echo "$CK" | wc -l)
for c in $(echo "$CK" | awk -v n=$N 'NR%5==0 || NR==n || NR==1'); do
  s=$(basename $c .safetensors | cut -d_ -f2)
  python scripts/evaluate_policy.py evaluation.checkpoint=$c evaluation.world_count=512 \
    evaluation.metrics_output=$O/eval/metrics_$s.json || echo "eval $s failed"
done
FIRST=$(echo "$CK" | head -1); LAST=$(echo "$CK" | tail -1)
for c in $FIRST $LAST; do
  s=$(basename $c .safetensors | cut -d_ -f2)
  python scripts/view_policy.py evaluation.checkpoint=$c viewer.backend=mp4 viewer.headless=true viewer.realtime=false \
    viewer.output_path=$O/video/policy_$s.mp4 viewer.num_frames=1000 viewer.video_fps=50 || echo "video $s failed"
done
ls -la $O/eval $O/video
