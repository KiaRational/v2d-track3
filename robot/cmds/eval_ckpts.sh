# Unassisted evaluation of the checkpoints saved so far (ep39 training in progress).
O=/scratch/kghasemz/v2d/robot/fc_out/ep39_dustpan; mkdir -p $O/eval
for c in $(ls $O/policy_*.safetensors | sort -t_ -k2 -n); do
  s=$(basename $c .safetensors | cut -d_ -f2); [ -f $O/eval/metrics_$s.json ] && continue
  python scripts/evaluate_policy.py evaluation.checkpoint=$c evaluation.world_count=512 \
    evaluation.metrics_output=$O/eval/metrics_$s.json 2>&1 | grep -v "^Module\|nefc" | tail -15
done
