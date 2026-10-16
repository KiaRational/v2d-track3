# Canonical Sharpa FlashSAC recipe on one of our sequences. Env (via APPTAINERENV_*): SEQ, NAME, CURR (curriculum, default
# tracking_250m_mixed_reset). 2048 worlds + njmax/nconmax 1024/512 (4096 worlds OOM with the larger contact buffers on an
# L40S; the default 512/256 overflows on thin objects). Offline wandb for the return curve.
P=/scratch/kghasemz/v2d/robot/hmd/ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave
export WANDB_MODE=offline WANDB_DIR=/scratch/kghasemz/v2d/robot/wandb
SUP=true; [ -s /scratch/kghasemz/v2d/robot/hmd/ego_recon/reconstructed_stage/${SEQ}_support.usda ] && \
  grep -qE "Cylinder|Cube" /scratch/kghasemz/v2d/robot/hmd/ego_recon/reconstructed_stage/${SEQ}_support.usda || SUP=false
python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${P}'" reset.seed=42 \
  curriculum=${CURR:-tracking_250m_mixed_reset} scene.support=$SUP sim.njmax=1024 sim.nconmax=512 scene.world_count=2048 \
  logging.mode=offline logging.upload_checkpoints=false logging.run_name=${NAME} \
  output_dir=/scratch/kghasemz/v2d/robot/fc_out/${NAME}
