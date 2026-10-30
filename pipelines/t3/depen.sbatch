#!/bin/bash
#SBATCH --job-name=depen
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# hand_depen.py (hands_c vs tpfsu_recon objects -> stereo/hands_dp) for all 40
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon OMP_NUM_THREADS=1
cd ~/v2d-vulcan/stereo
cat /scratch/kghasemz/v2d/beat1/runs40.txt | xargs -P 20 -I{} sh -c "python hand_depen.py {} 2>&1 | grep -v Warning" > /scratch/kghasemz/v2d/beat1/diag/depen.log
grep -c . /scratch/kghasemz/v2d/beat1/diag/depen.log
