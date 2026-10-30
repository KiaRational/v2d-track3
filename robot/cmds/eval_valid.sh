# Re-evaluate a run until every world is valid (environment_valid all true). The evaluator is deterministic (no seed),
# but a rare GPU-physics blow-up in one world depends on the batch: try world counts ${COUNTS:-512 256 128} on the last
# checkpoint, then on the previous one. Output eval/object_trajectories_<step>_w<count>.parquet (+ metrics).
# Env (via APPTAINERENV_*): NAME, optional COUNTS.
O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/eval
CK=$(ls $O/policy_*.safetensors | sort -V | tail -2 | tac)
for C in $CK; do
  s=$(basename $C .safetensors | cut -d_ -f2)
  for W in ${COUNTS:-512 256 128}; do
    T=$O/eval/object_trajectories_${s}_w${W}.parquet
    python scripts/evaluate_policy.py evaluation.checkpoint=$C evaluation.world_count=$W \
      evaluation.metrics_output=$O/eval/metrics_${s}_w${W}.json evaluation.object_trajectories_output=$T 2>&1 | grep -vE "^\s*$" | tail -3
    ok=$(/opt/venv/bin/python -c "import sys,pyarrow.parquet as pq; t=pq.read_table(sys.argv[1],columns=['environment_valid']); print(int(all(t.column(0).to_pylist())))" $T 2>&1 | tail -1)
    echo "$NAME ckpt $s worlds $W all_valid=$ok"
    [ "$ok" = 1 ] && { echo "VALID $T"; exit 0; }
    rm -f $T   # never leave an invalid rollout where the packer could pick it
  done
done
echo "NO VALID ROLLOUT for $NAME"
