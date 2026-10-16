#!/bin/bash
#SBATCH --job-name=v2d-graspfull
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Physically plausible grasps, wrench-first, one run per task (the main pipeline):
#   s06h candidates: squeeze 1.5 / 3 / 5 mm (contacts within 15 mm) + "close" (squeeze 1.5 mm,
#   every region within 30 mm pulled onto the surface)
#   -> contact-wrench feasibility (primary) and MuJoCo hold test (secondary) of each
#   -> s06i per-grasp choice -> final s06h pass measured against the stereo fit mano_fit
#   (trust limits, data term, anchors; each grasp keeps its selected squeeze/contact distance)
#   -> mano_fit_phys -> verification -> 4D video.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
R=$V2D_ROOT/runs/$NAME/stereo
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
$D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --squeeze_mm 1.5 --contact_mm 30 --out mano_fit_close
C="mano_fit_s15 mano_fit_close mano_fit_s30 mano_fit_s50"
$D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands mano_fit
for H in $C; do $D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands $H; done
for H in mano_fit $C; do MJ "$NAME" --variant "$VARIANT" --hands $H --stability; done
python s06i_select_grasps.py "$NAME" --cands $C --out mano_fit_selected
# Final pass: recompute MANO geometry and smooth candidate changes against exactly the same objects,
# measured against the original stereo fit (not the selected hands) and with the selected squeezes.
$D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --ref_hands mano_fit \
  --hands mano_fit_selected --segment_params mano_fit_selected --out mano_fit_phys
$D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands mano_fit_phys
MJ "$NAME" --variant "$VARIANT" --hands mano_fit_phys --stability
$D /code/stereo/audit_contact.py "$NAME" --variant "$VARIANT" --hands mano_fit
$D /code/stereo/audit_contact.py "$NAME" --variant "$VARIANT" --hands mano_fit_phys
$D /code/stages/render_4d_scene.py "$NAME" --objects "$VARIANT" --hands mano_fit_phys
