# Official evaluator on ONE chosen checkpoint (+ trajectory export) into eval_best/. Env: NAME, STEP, optional WORLDS.
O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/eval_best
C=$O/policy_${STEP}.safetensors; [ -f $C ] || { echo "missing $C"; exit 1; }
python scripts/evaluate_policy.py evaluation.checkpoint=$C evaluation.world_count=${WORLDS:-512} \
  evaluation.metrics_output=$O/eval_best/metrics_${STEP}.json \
  evaluation.object_trajectories_output=$O/eval_best/object_trajectories_${STEP}.parquet \
  2>&1 | grep -E "report|Error|add_auc" | tail -5
