#!/bin/bash
#SBATCH --job-name=tp-all
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# per run: track_pose (c14) -> tp_recon; hand_graph on it -> tp_hg_recon; pose_filter -> tpf_recon / tphgf_recon
set -uo pipefail
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon OMP_NUM_THREADS=2
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)
cd ~/v2d-vulcan/stereo
G=report; [ -f $V2D_ROOT/runs/$RUN/grasp_vlm.json ] && G=vlm
python track_pose.py $RUN 2>&1 | grep --line-buffered -v Warning | grep "{"
python hand_graph.py $RUN --src tp_recon --out tp_hg_recon --grasp $G 2>&1 | grep --line-buffered -v Warning
python pose_filter.py $RUN --src tp_recon --out tpf_recon 2>&1 | grep --line-buffered -v Warning
python pose_filter.py $RUN --src tp_hg_recon --out tphgf_recon 2>&1 | grep --line-buffered -v Warning
