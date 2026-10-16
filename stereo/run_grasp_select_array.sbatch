#!/bin/bash
#SBATCH --job-name=v2d-graspsel
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Selection part of the grasp loop only (candidates mano_fit_s15/s30/s50 already built by s06h
# against the same $V2D_OBJECT_VARIANT): wrench feasibility + stability of each -> per-grasp choice
# -> final s06h pass against the stereo fit (selected squeezes) -> verification -> 4D video.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
VARIANT=${V2D_OBJECT_VARIANT:-fused_all_sam3}
cd ~/v2d-vulcan/stereo
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python"
SIF=/scratch/kghasemz/franka/containers/python312_trixie.sif
MJ() { apptainer exec -B /scratch -B $HOME/v2d-vulcan "$SIF" bash -c \
  ". /scratch/kghasemz/v2d/venv-mj/bin/activate && V2D_ROOT=$V2D_ROOT python $HOME/v2d-vulcan/stereo/physics_check.py $*"; }
C="mano_fit_s15 mano_fit_s30 mano_fit_s50"
for H in $C; do
  V=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['args']['variant'])" "$V2D_ROOT/runs/$NAME/stereo/$H/report.json")
  [ "$V" = "$VARIANT" ] || { echo "$H was refined against $V, not $VARIANT; rebuild candidates" >&2; exit 1; }
done
for H in mano_fit $C; do $D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands $H; done
for H in mano_fit $C; do MJ "$NAME" --variant "$VARIANT" --hands $H --stability; done
python s06i_select_grasps.py "$NAME" --cands $C --out mano_fit_selected
$D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --ref_hands mano_fit \
  --hands mano_fit_selected --segment_params mano_fit_selected --out mano_fit_phys
$D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands mano_fit_phys
MJ "$NAME" --variant "$VARIANT" --hands mano_fit_phys --stability
MJ "$NAME" --variant "$VARIANT" --hands mano_fit_phys
$D /code/stereo/audit_contact.py "$NAME" --variant "$VARIANT" --hands mano_fit_phys
$D /code/stages/render_4d_scene.py "$NAME" --objects "$VARIANT" --hands mano_fit_phys
