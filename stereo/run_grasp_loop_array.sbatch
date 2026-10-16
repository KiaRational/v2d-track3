#!/bin/bash
#SBATCH --job-name=v2d-grasploop
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# ABLATION of run_grasp_full_array.sbatch (the main pipeline): same loop without the "close"
# candidate. Simulate -> diagnose -> correct, one run per task:
#   s06h candidates (squeeze 1.5 / 3 / 5 mm, contacts anchored at the observed places)
#   -> contact-wrench feasibility + MuJoCo per-frame grasp stability of each -> s06i per-grasp choice
#   -> final s06h pass against the stereo fit (selected squeezes) -> mano_fit_phys
#   -> verification (wrench, stability, penetration audit, replay) -> 4D video.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
VARIANT=${V2D_OBJECT_VARIANT:-fused_all_sam3}
cd ~/v2d-vulcan/stereo
[ -f "$V2D_ROOT/runs/$NAME/table_plane.json" ] || python table_plane.py "$NAME"
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python"
SIF=/scratch/kghasemz/franka/containers/python312_trixie.sif
MJ() { apptainer exec -B /scratch -B $HOME/v2d-vulcan "$SIF" bash -c \
  ". /scratch/kghasemz/v2d/venv-mj/bin/activate && V2D_ROOT=$V2D_ROOT python $HOME/v2d-vulcan/stereo/physics_check.py $*"; }
for S in 15 30 50; do
  $D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --squeeze_mm $(echo "$S / 10" | bc -l) --out mano_fit_s$S
done
C="mano_fit_s15 mano_fit_s30 mano_fit_s50"
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
