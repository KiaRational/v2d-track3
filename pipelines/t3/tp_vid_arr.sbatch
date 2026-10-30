#!/bin/bash
#SBATCH --job-name=tp-vid
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# c3 vs c14+filter (tpf_recon) side by side -> viz/c17/<run>_c3_vs_tpf.mp4
set -uo pipefail
source ~/v2d-vulcan/env.sh
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)
AG() { docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ \
  -e PYTHONPATH=/code/hoi4d/src:/code/stereo -e V2D_OBJ_DIR=objects_recon -e V2D_MESH_DIR=meshes_recon "$@"; }
O=$V2D_VIZ/c17; D=$V2D_VIZ/09_scene4d; mkdir -p $O
[ -f $D/${RUN}_c3bfix.mp4 ] || AG agile python /code/stages/render_4d_scene.py $RUN --objects fused_bfix_recon --obj_dir objects_recon --hands hands_c --no_gt --label "c3" --out_name ${RUN}_c3bfix.mp4 2>&1 | tail -1
AG agile python /code/stages/render_4d_scene.py $RUN --objects tpf_recon --obj_dir objects_recon --hands hands_c --no_gt \
  --label "c14 + filter (FP candidates + masks + tracks)" --out_name ${RUN}_tpf.mp4 2>&1 | tail -1
AG agile python /code/stereo/sbs_video.py $O/${RUN}_c3_vs_tpf.mp4 "$D/${RUN}_c3bfix.mp4:c3" "$D/${RUN}_tpf.mp4:c14 + filter" --height 480 2>&1 | tail -1
