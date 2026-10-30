#!/bin/bash
#SBATCH --job-name=t6-carve
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# carve the dish rack's collision where another object really sits in it (t6 tree), then release the held t6 trainings
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon
T=/scratch/kghasemz/v2d/robot/hmd_t6_fine
for x in "evaluation_ep000030 mini_sweeper" "evaluation_ep000015 wooden_spoon" "evaluation_ep000016 wooden_spoon" "evaluation_ep000014 blue_cup" "evaluation_ep000028 white_pot"; do
  set -- $x; python ~/v2d-vulcan/robot/carve_collision.py $T $1 plastic_dish_rack $2 2>&1 | grep -v Warning
done
grep -c "<collision>" $T/ego_recon/processed/plastic_dish_rack_rigid.urdf.precarve $T/ego_recon/processed/plastic_dish_rack_rigid.urdf
for j in $(squeue -u $USER -h -t PD -o "%i %j %r" | awk '$2=="t6-sac" && $3=="JobHeldUser" {print $1}'); do scontrol release $j; done
echo "released t6 trainings"
