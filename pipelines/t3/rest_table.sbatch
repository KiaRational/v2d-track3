#!/bin/bash
#SBATCH --job-name=rest-table
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# rest_table.py tpfsu_recon -> tpfsut_recon for all 40
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon OMP_NUM_THREADS=1
cd ~/v2d-vulcan/stereo
cat /scratch/kghasemz/v2d/beat1/runs40.txt | xargs -P 20 -I{} sh -c "python rest_table.py {} 2>&1 | grep -v Warning" > /scratch/kghasemz/v2d/beat1/diag/rest_table.log
c=0; for r in $(cat /scratch/kghasemz/v2d/beat1/runs40.txt); do [ -f $V2D_ROOT/runs/$r/tpfsut_recon/rest_table_report.json ] && c=$((c+1)); done; echo "done $c/40"
