# Re-evaluate a run until every world is valid (environment_valid all true): last checkpoint with reset seeds
# ${SEEDS:-7 13}, then the previous checkpoint. Output eval/object_trajectories_<step>_s<seed>.parquet (+ metrics).
# Env (via APPTAINERENV_*): NAME, optional WORLDS (default 512), SEEDS.
O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/eval
CK=$(ls $O/policy_*.safetensors | sort -V | tail -2 | tac)
for C in $CK; do
  s=$(basename $C .safetensors | cut -d_ -f2)
  for SEED in ${SEEDS:-7 13}; do
    T=$O/eval/object_trajectories_${s}_s${SEED}.parquet
    python scripts/evaluate_policy.py evaluation.checkpoint=$C evaluation.world_count=${WORLDS:-512} reset.seed=$SEED \
      evaluation.metrics_output=$O/eval/metrics_${s}_s${SEED}.json evaluation.object_trajectories_output=$T 2>&1 | grep -vE "^\s*$" | tail -5
    ok=$(/opt/venv/bin/python -c "import sys,pyarrow.parquet as pq; t=pq.read_table(sys.argv[1],columns=['environment_valid']); print(int(all(t.column(0).to_pylist())))" $T 2>/dev/null)
    echo "$NAME ckpt $s seed $SEED all_valid=$ok"
    [ "$ok" = 1 ] && { echo "VALID $T"; exit 0; }
  done
done
echo "NO VALID ROLLOUT for $NAME"
