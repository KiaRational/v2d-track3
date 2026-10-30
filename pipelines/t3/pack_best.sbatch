#!/bin/bash
#SBATCH --job-name=best-pack
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Track 3 pack + preflight from a per-episode selection: each scored episode takes its rollout from the chosen run
# (newest rollout of that run whose worlds are ALL valid) and its meshes from that run's simulator tree
# (robot/<tree>/ego_recon/processed/<obj>_visual.stl: the body frame of its rollout). NO upload.
# Usage: sbatch pack_best.sbatch <out_dir> <commit_url> <selection.txt>   (lines: "<ep 2-digit> <run> <tree>")
set -uo pipefail
module load python/3.11.5 arrow/24.0.0
source /scratch/kghasemz/v2d/venv-kaggle/bin/activate
V=/scratch/kghasemz/v2d; KIT=$V/submission_kit_new/unz/v2d_submission_kit
OUT=$1; COMMIT=$2; SEL=$3
mkdir -p "$OUT"; rm -rf "${OUT:?}/results" "${OUT:?}/meshes"; mkdir -p "$OUT/results" "$OUT/meshes"
jq -r '.episodes | to_entries[] | "\(.key) \(.value|join(" "))"' $KIT/data/track_3_evaluation_objects.json | while read -r e objs; do
  n=$(printf "%02d" $e); E=episode_$(printf %06d $e)
  read -r RUNNAME TREE <<< "$(awk -v e=$n '$1==e {print $2, $3}' $SEL)"
  P=$V/robot/$TREE/ego_recon/processed
  t=""
  for c in $(ls -t $V/robot/fc_out/$RUNNAME/eval/object_trajectories_*.parquet 2>/dev/null); do
    ok=$(python -c "import sys,pyarrow.parquet as pq; t=pq.read_table(sys.argv[1],columns=['environment_valid']); print(int(all(t.column(0).to_pylist())))" $c)
    [ "$ok" = 1 ] && { t=$c; break; }
  done
  [ -n "$t" ] || { echo "NO VALID rollout e$n ($RUNNAME)"; continue; }
  ln -s "$t" "$OUT/results/$E.parquet"; echo "e$n <- $RUNNAME/$(basename $t), meshes $TREE"
  mkdir -p "$OUT/meshes/$E"
  for o in $objs; do
    [ -f $P/${o}_visual.stl ] || { echo "MISSING mesh $o ($TREE)"; continue; }
    ln -s $P/${o}_visual.stl "$OUT/meshes/$E/$o.stl"
  done
done
cd "$KIT"
echo "=== pack"
python tools/pack_submission.py --track track_3 --results "$OUT/results" --meshes "$OUT/meshes" \
  --code-commit "$COMMIT" --eval-script $KIT/flash_chord/scripts/evaluate_policy.py \
  -o "$OUT/track3_submission.parquet"; echo "pack exit $?"
echo "=== preflight"
python tools/preflight.py --track track_3 --results "$OUT/results"; echo "preflight exit $?"
ls -la "$OUT"
