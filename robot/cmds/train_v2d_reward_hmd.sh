# = train_v2d_reward.sh with HMD (env) selecting the motion-data tree, e.g. robot/hmd_phys (realistic object mass).
# V2D-reward Sharpa FlashSAC recipe (experiment=sharpa_flash_sac_v2d: object-dominant tracking, human contact
# locations in the object frame with inside/outside disambiguation, smoothness, completion bonus) on one of our
# sequences. Same env-variable interface as train_generic.sh: SEQ, NAME (via APPTAINERENV_*), optional CURR
# (default tracking_250m_v2d; the v2d curriculum must list the V2D term weights, the canonical ones do not).
# Optional EXTRA: further Hydra overrides (e.g. objective.object_orientation.var=0.1).
# Needs the v2d-rewards worktree mounted as the FlashCHORD source:
#   FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=ep12_bluecup APPTAINERENV_NAME=ep12_v2d \
#     sbatch --time=4:00:00 /home/kghasemz/v2d-vulcan/robot/run_fc.sbatch /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward.sh
# 2048 worlds + njmax/nconmax 1024/512 as in train_generic.sh; 250M steps (experiment default). Checkpoints keep a
# canonical-objective resolved config, so eval_generic.sh / the official evaluator work with the unmodified fc_src.
if [ ! -f /workspace/src/flash_chord/configs/experiment/sharpa_flash_sac_v2d.yaml ]; then
  echo "ERROR: /workspace is not the v2d-rewards worktree; submit with FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards" >&2
  exit 2
fi
P=${HMD:-/scratch/kghasemz/v2d/robot/hmd}/ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave
export WANDB_MODE=offline WANDB_DIR=/scratch/kghasemz/v2d/robot/wandb
SUP=true; [ -s /scratch/kghasemz/v2d/robot/hmd/ego_recon/reconstructed_stage/${SEQ}_support.usda ] && \
  grep -qE "Cylinder|Cube" /scratch/kghasemz/v2d/robot/hmd/ego_recon/reconstructed_stage/${SEQ}_support.usda || SUP=false
# Our reconstructed table (write_table_support.py, prim "table"): the finite table replaces the infinite ground plane
# at table height, and the hands collide with it (physical; fingers can still reach under the rim). TABLE=0 opts out.
TBL=""; [ "${TABLE:-1}" = 0 ] && SUP=false  # ground plane + our table together launches objects (overlapping supports)
if [ "$SUP" = true ] && [ "${TABLE:-1}" = 1 ] && \
   grep -q '"table"' /scratch/kghasemz/v2d/robot/hmd/ego_recon/reconstructed_stage/${SEQ}_support.usda; then
  TBL="collision.ground=false collision.robot_support_collision=true"
fi
echo "support=$SUP table overrides: ${TBL:-none}"
python scripts/train_flash_sac.py experiment=sharpa_flash_sac_v2d "task.parquet='${P}'" reset.seed=42 \
  curriculum=${CURR:-tracking_250m_v2d} scene.support=$SUP ${TBL} sim.njmax=1024 sim.nconmax=512 scene.world_count=2048 \
  logging.mode=offline logging.upload_checkpoints=false logging.run_name=${NAME} \
  ${EXTRA} output_dir=/scratch/kghasemz/v2d/robot/fc_out/${NAME}
