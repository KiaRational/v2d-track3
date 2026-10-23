#!/bin/bash
#SBATCH --job-name=v2d-convert
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Our reconstruction -> FlashCHORD training data, with NVIDIA's own pipeline:
#   export result bundle (ours) -> ego_recon loader -> rigid URDF -> retarget to Sharpa -> support surface
# Usage: sbatch run_convert.sbatch <run> <object> [seq_name]
#        sbatch run_convert.sbatch <run> all|A,B [seq_name]   (multi-object: every object of the episode, or
#        that subset, in the reference object-slot order; one rigid body per object)
# Export warns (manifest quality.passed=false) on a failed hand quality check; V2D_EXPORT_STRICT=1 refuses.
# Resting objects are settled flat on the table (rest_settle.py; V2D_EXPORT_EXTRA=--no_rest_settle to skip).
# V2D_HANDS picks the hand variant (default mano_fit_phys), V2D_OBJECT_VARIANT the objects (fused_all_sam3).
# The loader comes from fc_src (patches/ego_recon_loader.py: get_fps from the bundle, multi-object bundles):
# the rg image has its own upstream copy in site-packages, so fc_src's lib dir is bind-mounted over it.
set -euo pipefail
source ~/v2d-vulcan/env.sh
RUN=$1; OBJ=$2; SEQ=${3:-${RUN}_${OBJ//,/_}}
CLIP=$V2D_ROOT/robot/clips/$SEQ
SEL=(--object "$OBJ"); [[ $OBJ == all || $OBJ == *,* ]] && SEL=(--objects_list "$OBJ")
python ~/v2d-vulcan/robot/export_result_bundle.py "$RUN" "${SEL[@]}" --hands "${V2D_HANDS:-mano_fit_phys}" \
  --objects "${V2D_OBJECT_VARIANT:-fused_all_sam3}" --out $CLIP/outputs/result ${V2D_EXPORT_STRICT:+--strict} ${V2D_EXPORT_EXTRA:-}
OBJ=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['sources']['object'])" $CLIP/outputs/result/manifest.json)
# keep the sim ground at our table top: cancel the loader's frame-0 'lowest object point -> z = 0' shift
ZOFF=$(python -c "import json, sys; print(json.load(open(sys.argv[1])).get('loader_ground_z_offset', 0.0))" $CLIP/outputs/result/manifest.json)
HMD=$V2D_ROOT/robot/hmd; mkdir -p $HMD $V2D_ROOT/robot/rg_cache
RG=$V2D_ROOT/fc_src/robotic_grounding
LIB=$V2D_ROOT/fc_src/reconstruction/modules/v2d_task_library_loader/lib
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:$HOME/v2d-vulcan -e HUMAN_MOTION_DATA_DIR=$HMD \
  -v $LIB:/opt/conda/lib/python3.11/site-packages/v2d/task_library_loader/lib \
  -e ROBOTIC_GROUNDING_INTERMEDIATE_DIR=$V2D_ROOT/robot/rg_cache \
  -e PYTHONPATH=$RG/source/robotic_grounding -e V2D_ROOT=$V2D_ROOT rg bash -c "
set -e; cd $RG
python -c 'import v2d.task_library_loader.lib.ego_recon_loader as m; assert \"_bodies\" in open(m.__file__).read(), m.__file__'
python -m v2d.task_library_loader.lib.run_loader --dataset ego_recon --dataset_root $CLIP --object_name $OBJ \
  --sequence_name $SEQ --no_ground_align --ground_z_offset=$ZOFF --mano_model_dir $V2D_ROOT/weights/hand --device cuda:0 --save
python scripts/generate_rigid_urdfs.py --dataset ego_recon
python scripts/retarget/run_retarget.py --dataset ego_recon --robot sharpa_wave --sequence_pattern '$SEQ' --save
python scripts/reconstruct_support_surfaces.py --dataset ego_recon --sequence_pattern '$SEQ' || echo 'support surface step failed (continuing)'
# our round table as the sim support surface (finite, so hands can go under the rim); after upstream's step
python /home/kghasemz/v2d-vulcan/robot/write_table_support.py $CLIP '$SEQ' --zoff=$ZOFF
python /home/kghasemz/v2d-vulcan/robot/write_collision_mesh.py $CLIP --objects=${V2D_PROVIDED_COLLISION:-all}
find $HMD -name '*.parquet' | grep -i '$SEQ' | head
"
