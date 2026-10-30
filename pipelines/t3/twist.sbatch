#!/bin/bash
#SBATCH --job-name=twist
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# twist_unwrap.py on c19 (tpfs_recon -> tpfsu_recon) for all 40
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon OMP_NUM_THREADS=1
cd ~/v2d-vulcan/stereo
cat /scratch/kghasemz/v2d/beat1/runs40.txt | xargs -P 20 -I{} sh -c "python twist_unwrap.py {} 2>&1 | grep -v Warning"
c=0; for r in $(cat /scratch/kghasemz/v2d/beat1/runs40.txt); do [ -f $V2D_ROOT/runs/$r/tpfsu_recon/twist_unwrap_report.json ] && c=$((c+1)); done; echo "done $c/40"
