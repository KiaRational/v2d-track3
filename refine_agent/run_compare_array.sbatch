#!/bin/bash
#SBATCH --job-name=v2d-agentviz
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Before/after-agent 4D comparison video for one episode per task -> $V2D_VIZ/12_agent/<run>_agent_compare.mp4
# (top: before, fused_all_sam3; bottom: after, fused_agent with a banner on every fixed frame; green = mocap)
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
H=${HANDS:-mano_fit_phys}
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python /code/stages/render_4d_scene.py"
$D "$NAME" --objects fused_all_sam3 --hands $H --label "BEFORE agent" --out_name ${NAME}_before_agent.mp4
$D "$NAME" --objects fused_agent --hands $H --label "AFTER agent" --out_name ${NAME}_after_agent.mp4 \
   --agent_log $V2D_ROOT/runs/$NAME/refine/agent_log.json
mkdir -p $V2D_VIZ/12_agent
ffmpeg -loglevel error -y -i $V2D_VIZ/09_scene4d/${NAME}_before_agent.mp4 -i $V2D_VIZ/09_scene4d/${NAME}_after_agent.mp4 \
  -filter_complex vstack=inputs=2 -c:v libx264 -crf 22 -pix_fmt yuv420p $V2D_VIZ/12_agent/${NAME}_agent_compare.mp4
rm -f $V2D_VIZ/09_scene4d/${NAME}_before_agent.mp4 $V2D_VIZ/09_scene4d/${NAME}_after_agent.mp4
ls -lh $V2D_VIZ/12_agent/${NAME}_agent_compare.mp4
