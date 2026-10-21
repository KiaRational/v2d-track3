# Recipe v3/v4 (login-node driver, sbatch only): for every run in the list, the run's hoi4d reconstruction (its
# versions.json _current after `refine --version initial --name <TAG>`) -> convert -> parquet re-pointed at
# robot/hmd_v3 (realistic mass, spoon 0.1 kg, inertia floor 1e-4) -> train (FlashSAC tracking_250m_abl_contact,
# real-time playback) -> eval (+ video) -> official eval of the last checkpoint with the trajectory export.
# Usage: TAG=v4 bash submit_v3.sh <runs.txt>      Seqs <TAG>_eNN / <TAG>_pNN, runs fc_out/<seq>_rt.
# Two passes: ALL conversions first, one at a time (they rewrite shared object meshes in hmd/ego_recon/processed;
# concurrent conversions truncated an STL a training was loading), then the trainings, which wait for the last one.
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
TAG=${TAG:-v3}; LOG=/scratch/kghasemz/v2d/beat1/${TAG}_runs.tsv
FPS=20.254119873046875; EXTRA="sim.fps=$FPS task.motion_speed=1.0"; HMDV3=/scratch/kghasemz/v2d/robot/hmd_v3
KITJ=/scratch/kghasemz/v2d/submission_kit_new/unz/v2d_submission_kit/data/track_3_evaluation_objects.json
declare -A CONV PQ META
PREVC=
for R in $(cat "$1"); do
  REG=/scratch/kghasemz/v2d/runs/$R/hoi4d/versions.json
  CUR=$(jq -r '._current // "initial"' $REG); OV=$(jq -r --arg v "$CUR" '.[$v].objects_variant' $REG); HV=$(jq -r --arg v "$CUR" '.[$v].hands_variant' $REG)
  ep=$((10#${R##*_ep})); NN=$(printf %02d $ep)
  if [[ $R == evaluation_* ]]; then S=${TAG}_e$NN; OBJ=$(jq -r --arg e $ep '.episodes[$e] | join(",")' $KITJ); else S=${TAG}_p$NN; OBJ=all; fi
  c=$(V2D_HANDS=$HV V2D_OBJECT_VARIANT=$OV sbatch --parsable --exclude=rack02-06 ${PREVC:+--dependency=afterany:$PREVC} \
      --job-name=${TAG}-conv run_convert.sbatch $R $OBJ $S); PREVC=$c
  p=$(HMD_DST=$HMDV3 sbatch --parsable --exclude=rack02-06 --dependency=afterok:$c /scratch/kghasemz/v2d/beat1/phys_pq.sbatch $S)
  CONV[$R]=$c; PQ[$R]=$p; META[$R]="$S $CUR $OV $HV"
done
for R in $(cat "$1"); do
  read -r S CUR OV HV <<< "${META[$R]}"; N=${S}_rt; c=${CONV[$R]}; p=${PQ[$R]}
  t=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=$N APPTAINERENV_HMD=$HMDV3 APPTAINERENV_EXTRA="$EXTRA" \
      APPTAINERENV_CURR=tracking_250m_abl_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDV3 \
      sbatch --parsable --time=7:00:00 --exclude=rack02-06 --dependency=afterok:$p,afterany:$PREVC --job-name=${TAG}-train run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward_hmd.sh)
  e=$(APPTAINERENV_NAME=$N APPTAINERENV_CAMP=0.8,-1.1,0.8 APPTAINERENV_CAMT=0,0,0 APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDV3 \
      sbatch --parsable --time=2:00:00 --exclude=rack02-06 --dependency=afterany:$t --job-name=${TAG}-eval run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_generic.sh)
  f=$(APPTAINERENV_NAME=$N APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDV3 \
      sbatch --parsable --time=1:00:00 --exclude=rack02-06 --dependency=afterany:$e --job-name=${TAG}-evlast run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_last.sh)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $R $N $CUR $OV $HV $c $t $e $f | tee -a $LOG
done
