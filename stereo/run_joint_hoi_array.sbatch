#!/bin/bash
#SBATCH --job-name=v2d-jointhoi
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Joint hand-object optimisation, one run per task (sbatch --array=0-N run_joint_hoi_array.sbatch <list> [s06j args]):
#   s06j (fused_all_sam3 + mano_fit_phys -> fused_joint + mano_fit_joint) -> mocap eval (eval_joint.json)
#   -> contact audit + wrench feasibility, each hand set against ITS object variant
#   -> before/after 4D video $V2D_VIZ/14_joint/<run>_joint_compare.mp4 (top: before, bottom: joint)
set -euo pipefail
source ~/v2d-vulcan/env.sh
LIST=$1; shift
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
INIT=${V2D_OBJECT_VARIANT:-fused_all_sam3}
cd ~/v2d-vulcan/stereo
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ -e V2D_OBJ_DIR=objects_sam3 agile python"
T0=$(date +%s)
$D /code/stereo/s06j_joint_hoi.py "$NAME" --init_objects "$INIT" --variant fused_joint --out mano_fit_joint "$@"
echo "s06j wall time $(( $(date +%s) - T0 )) s"
export V2D_OBJ_DIR=objects_sam3
[[ $NAME == public_* ]] && python ../stages/eval_objects.py "$NAME" --variants "$INIT" fused_joint --align fp --out eval_joint.json
# baseline hands against the baseline objects, joint hands against the joint objects (outputs keyed by hands name)
$D /code/stereo/audit_contact.py "$NAME" --variant "$INIT" --hands mano_fit_phys
$D /code/stereo/audit_contact.py "$NAME" --variant fused_joint --hands mano_fit_joint
$D /code/stereo/wrench_feasibility.py "$NAME" --variant "$INIT" --hands mano_fit_phys
$D /code/stereo/wrench_feasibility.py "$NAME" --variant fused_joint --hands mano_fit_joint
R="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python /code/stages/render_4d_scene.py"
$R "$NAME" --objects "$INIT" --hands mano_fit_phys --label "BEFORE: $INIT + mano_fit_phys" --out_name ${NAME}_jbefore.mp4
$R "$NAME" --objects fused_joint --hands mano_fit_joint --label "JOINT: fused_joint + mano_fit_joint" --out_name ${NAME}_jafter.mp4
mkdir -p $V2D_VIZ/14_joint
ffmpeg -loglevel error -y -i $V2D_VIZ/09_scene4d/${NAME}_jbefore.mp4 -i $V2D_VIZ/09_scene4d/${NAME}_jafter.mp4 \
  -filter_complex vstack=inputs=2 -c:v libx264 -crf 22 -pix_fmt yuv420p $V2D_VIZ/14_joint/${NAME}_joint_compare.mp4
rm -f $V2D_VIZ/09_scene4d/${NAME}_jbefore.mp4 $V2D_VIZ/09_scene4d/${NAME}_jafter.mp4
ls -lh $V2D_VIZ/14_joint/${NAME}_joint_compare.mp4
# summary
RUN=$V2D_ROOT/runs/$NAME
[ -f $RUN/eval_joint.json ] && jq -c --arg a "$INIT" '.variants | to_entries[] | select(.key==$a or .key=="fused_joint") |
  {variant: .key, AUC: .value.official.AUC, per_object: (.value.objects | map_values({ADD_cm: .ADD_cm_mean, ADDS_cm: .ADDS_cm_mean, rot_deg: .rot_err_deg_mean, trans_cm: .trans_err_cm_mean}))}' $RUN/eval_joint.json
for H in mano_fit_phys mano_fit_joint; do
  jq -c --arg h $H '{hands: $h, audit: (.objects | map_values({pen_mm, pen_gt5, pen_gt10, gap_mm, tips_near}))}' $RUN/stereo/contact_audit_$H.json
  jq -c --arg h $H '{hands: $h, wrench: (.objects | map_values({lifted_grasp_frames, feasible_frac}))}' $RUN/stereo/wrench_$H.json
  jq -c --arg h $H '{hands: $h, quality_passed, q: (.quality_summary | map_values({passed, obj_p95: .object.p95_mm, tab_p95: .table.p95_mm}))}' $RUN/stereo/$H/report.json
done
jq -c '{runtime_s, objects: (.objects | map_values({centroid_change_mm, rot_change_deg})), hands}' $RUN/fused_joint/joint_report.json
