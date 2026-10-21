# Recipe v3 (login-node driver, sbatch only): for every run in the list, the hoi4d v3 reconstruction (the run's
# versions.json _current after `refine --version initial --name v3`: rest consistency + hand spin repair +
# TrackEverything-calibrated refine) -> convert -> parquet re-pointed at robot/hmd_v3 (realistic mass, spoon 0.1 kg,
# inertia floor 1e-4) -> train (FlashSAC tracking_250m_abl_contact, real-time playback) -> eval (+ video) -> official
# eval of the last checkpoint with the trajectory export. Usage: bash submit_v3.sh <runs.txt>   Seqs v3_eNN / v3_pNN.
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
FPS=20.254119873046875; EXTRA="sim.fps=$FPS task.motion_speed=1.0"; HMDV3=/scratch/kghasemz/v2d/robot/hmd_v3
KITJ=/scratch/kghasemz/v2d/submission_kit_new/unz/v2d_submission_kit/data/track_3_evaluation_objects.json
LOG=/scratch/kghasemz/v2d/beat1/v3_runs.tsv
for R in $(cat "$1"); do
  REG=/scratch/kghasemz/v2d/runs/$R/hoi4d/versions.json
  CUR=$(jq -r '._current // "initial"' $REG); OV=$(jq -r --arg v "$CUR" '.[$v].objects_variant' $REG); HV=$(jq -r --arg v "$CUR" '.[$v].hands_variant' $REG)
  ep=$((10#${R##*_ep})); NN=$(printf %02d $ep)
  if [[ $R == evaluation_* ]]; then S=v3_e$NN; OBJ=$(jq -r --arg e $ep '.episodes[$e] | join(",")' $KITJ); else S=v3_p$NN; OBJ=all; fi
  N=${S}_rt
  c=$(V2D_HANDS=$HV V2D_OBJECT_VARIANT=$OV sbatch --parsable --exclude=rack02-06 --job-name=v3-conv run_convert.sbatch $R $OBJ $S)
  p=$(HMD_DST=$HMDV3 sbatch --parsable --exclude=rack02-06 --dependency=afterok:$c /scratch/kghasemz/v2d/beat1/phys_pq.sbatch $S)
  t=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=$N APPTAINERENV_HMD=$HMDV3 APPTAINERENV_EXTRA="$EXTRA" \
      APPTAINERENV_CURR=tracking_250m_abl_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDV3 \
      sbatch --parsable --time=7:00:00 --exclude=rack02-06 --dependency=afterok:$p --job-name=v3-train run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward_hmd.sh)
  e=$(APPTAINERENV_NAME=$N APPTAINERENV_CAMP=0.8,-1.1,0.8 APPTAINERENV_CAMT=0,0,0 APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDV3 \
      sbatch --parsable --time=2:00:00 --exclude=rack02-06 --dependency=afterany:$t --job-name=v3-eval run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_generic.sh)
  f=$(APPTAINERENV_NAME=$N APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDV3 \
      sbatch --parsable --time=1:00:00 --exclude=rack02-06 --dependency=afterany:$e --job-name=v3-evlast run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_last.sh)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $R $N $CUR $OV $HV $c $t $e $f | tee -a $LOG
done
