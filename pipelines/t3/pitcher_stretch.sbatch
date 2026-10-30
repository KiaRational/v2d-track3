#!/bin/bash
#SBATCH --job-name=pitcher-stretch
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# lengthen the pitcher mesh at the bottom (extend_mesh.py), then redo the table step for the pitcher episodes
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon
cd ~/v2d-vulcan/stereo
python extend_mesh.py water_pitcher evaluation_ep000020 public_ep000021 public_ep000022 --dry 2>&1 | grep -v Warning
[ "${DRY:-0}" = 1 ] && exit 0
python extend_mesh.py water_pitcher evaluation_ep000020 public_ep000021 public_ep000022 2>&1 | grep -v Warning
for r in evaluation_ep000020 public_ep000021 public_ep000022; do python rest_table.py $r 2>&1 | grep -v Warning | grep water_pitcher; done
