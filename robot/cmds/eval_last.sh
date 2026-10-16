# Unassisted evaluation of the LAST checkpoint of a run with the official evaluator, plus its per-slot object
# trajectory export (the Track 3 submission input). Env (via APPTAINERENV_*): NAME, optional WORLDS (default 512).
# Run with the unmodified fc_src (no FC_SRC):  APPTAINERENV_NAME=<run> sbatch run_fc.sbatch cmds/eval_last.sh
O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/eval
LAST=$(ls $O/policy_*.safetensors | sort -V | tail -1); s=$(basename $LAST .safetensors | cut -d_ -f2)
python scripts/evaluate_policy.py evaluation.checkpoint=$LAST evaluation.world_count=${WORLDS:-512} \
  evaluation.metrics_output=$O/eval/metrics_$s.json evaluation.object_trajectories_output=$O/eval/object_trajectories_$s.parquet \
  2>&1 | grep -vE "^\s*$" | tail -40
/opt/venv/bin/python - "$O/eval/object_trajectories_$s.parquet" <<'PY'
import sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
cols = t.column_names
print("trajectory parquet:", sys.argv[1], t.num_rows, "rows; columns:", cols)
for c in ("object_id", "object_body_id", "object_body_name"):
    if c in cols:
        print(c, sorted(set(map(str, t.column(c).to_pylist()))))
b = t.column("object_body_id").to_numpy(); n = t.column("object_body_name").to_pylist()
print("slot -> name:", sorted({(int(i), str(x)) for i, x in zip(b[:1000], n[:1000])}))
md = t.schema.metadata or {}
print("metadata keys:", [k.decode() for k in md])
PY
