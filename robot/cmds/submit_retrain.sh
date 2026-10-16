# Login-node driver (sbatch only): retrain + eval existing converted sequences with extra Hydra overrides.
# Real-time playback for the Track-3 row template (1 row per video frame): RT=1 adds
#   sim.fps=<parquet fps, float32-exact> task.motion_speed=1.0  (one control step per source frame, no resampling;
#   the ManoSharpa loader does not support source_frame_playback). Checkpoints keep the config, so eval follows.
# Usage: SUFFIX=rt EXTRA="sim.fps=20.254119873046875 task.motion_speed=1.0" bash submit_retrain.sh hop_p00 hop_e01 ...
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
SUFFIX=${SUFFIX:?}; EXTRA=${EXTRA:?}
LOG=/scratch/kghasemz/v2d/robot/submit_${SUFFIX}.tsv
for S in "$@"; do
  N=${S}_${SUFFIX}
  t=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=$N APPTAINERENV_EXTRA="$EXTRA" \
      APPTAINERENV_CURR=tracking_250m_abl_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=5:00:00 --exclude=rack02-06 --job-name=rt-train run_fc.sbatch /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward.sh)
  e=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_NAME=$N APPTAINERENV_CAMP=0.8,-1.1,0.8 APPTAINERENV_CAMT=0,0,0 \
      APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=2:00:00 --exclude=rack02-06 --dependency=afterok:$t --job-name=rt-eval run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_generic.sh)
  f=$(APPTAINERENV_NAME=$N APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=1:00:00 --exclude=rack02-06 --dependency=afterok:$e --job-name=rt-evlast run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_last.sh)
  printf "%s\t%s\t%s\t%s\t%s\n" $S $N $t $e $f | tee -a $LOG
done
