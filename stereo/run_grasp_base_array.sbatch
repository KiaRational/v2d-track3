#!/bin/bash
#SBATCH --job-name=v2d-graspbase
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
B=${BASE:-mano_fit}  # base stereo hand fit (e.g. mano_fit_hop); outputs are ${B}_s15 ... ${B}_phys
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
  $D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --hands $B --squeeze_mm $(echo "$S / 10" | bc -l) --out ${B}_s$S
done
$D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --hands $B --squeeze_mm 1.5 --contact_mm 30 --out ${B}_close
C="${B}_s15 ${B}_close ${B}_s30 ${B}_s50"
$D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands $B
for H in $C; do $D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands $H; done
for H in $B $C; do MJ "$NAME" --variant "$VARIANT" --hands $H --stability; done
python s06i_select_grasps.py "$NAME" --cands $C --out ${B}_selected
# Final pass: recompute MANO geometry and smooth candidate changes against exactly the same objects,
# measured against the original stereo fit (not the selected hands) and with the selected squeezes.
$D /code/stereo/s06h_grasp_plausible.py "$NAME" --variant "$VARIANT" --ref_hands $B \
  --hands ${B}_selected --segment_params ${B}_selected --out ${B}_phys
$D /code/stereo/wrench_feasibility.py "$NAME" --variant "$VARIANT" --hands ${B}_phys
MJ "$NAME" --variant "$VARIANT" --hands ${B}_phys --stability
$D /code/stereo/audit_contact.py "$NAME" --variant "$VARIANT" --hands $B
$D /code/stereo/audit_contact.py "$NAME" --variant "$VARIANT" --hands ${B}_phys
$D /code/stages/render_4d_scene.py "$NAME" --objects "$VARIANT" --hands ${B}_phys --out_name ${NAME}_${B}_phys.mp4
