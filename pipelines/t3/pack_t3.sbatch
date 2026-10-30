#!/bin/bash
#SBATCH --job-name=t3-pack
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Track 3 pack + preflight for the t3 policies (poses tpf_recon, task2 reward). Meshes = OUR meshes exactly as the
# simulator used them (robot/hmd_t3 processed <obj>_visual.stl, body frame, zero visual origin). Results flat
# (results/episode_XXXXXX.parquet), newest rollout whose worlds are ALL valid. NO upload.
# Usage: sbatch pack_t3.sbatch <out_dir> <commit_url>
set -uo pipefail
module load python/3.11.5 arrow/24.0.0
source /scratch/kghasemz/v2d/venv-kaggle/bin/activate
V=/scratch/kghasemz/v2d; KIT=$V/submission_kit_new/unz/v2d_submission_kit; P=$V/robot/hmd_t3/ego_recon/processed
OUT=$1; COMMIT=$2
mkdir -p "$OUT"; rm -rf "${OUT:?}/results" "${OUT:?}/meshes"; mkdir -p "$OUT/results" "$OUT/meshes"
jq -r '.episodes | to_entries[] | "\(.key) \(.value|join(" "))"' $KIT/data/track_3_evaluation_objects.json | while read -r e objs; do
  n=$(printf "%02d" $e); E=episode_$(printf %06d $e)
  t=""
  for c in $(ls -t $V/robot/fc_out/t3_e${n}_task2/eval/object_trajectories_*.parquet 2>/dev/null); do
    ok=$(python -c "import sys,pyarrow.parquet as pq; t=pq.read_table(sys.argv[1],columns=['environment_valid']); print(int(all(t.column(0).to_pylist())))" $c)
    [ "$ok" = 1 ] && { t=$c; break; }
  done
  [ -n "$t" ] || { echo "NO VALID rollout e$n"; continue; }
  ln -s "$t" "$OUT/results/$E.parquet"; echo "e$n <- $(basename $t)"
  mkdir -p "$OUT/meshes/$E"
  for o in $objs; do
    [ -f $P/${o}_visual.stl ] || { echo "MISSING mesh $o"; continue; }
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
