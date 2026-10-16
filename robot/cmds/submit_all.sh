# Login-node driver (sbatch calls only): for every episode, convert -> train -> eval (every 20M ckpt + video) ->
# official eval of the last ckpt with the per-slot object trajectory export (submission input).
# Same recipe as ep23/24_depen_contact (tracking_250m_abl_contact, table + floor, depenetrate, rest settle).
# Usage: HANDS=mano_fit_hop_phys TAG=hop bash submit_all.sh public_list.txt eval_list.txt
# Seq names: ${TAG}_p<NN> (public) / ${TAG}_e<NN> (evaluation); runs fc_out/<seq>_contact. Job ids -> $LOG.
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
HANDS=${HANDS:?}; TAG=${TAG:?}
KITJ=/scratch/kghasemz/v2d/submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json
LOG=/scratch/kghasemz/v2d/robot/submit_${TAG}.tsv
for L in "$@"; do for R in $(cat "$L"); do
  ep=$((10#${R##*_ep})); NN=$(printf %02d $ep)
  if [[ $R == evaluation_* ]]; then
    S=${TAG}_e$NN; OBJ=$(jq -r --arg e $ep '.episodes[$e] | map(select(. != "wooden_spoon")) | join(",")' $KITJ)
  else
    S=${TAG}_p$NN; OBJ=all
  fi
  c=$(V2D_HANDS=$HANDS sbatch --parsable --job-name=${TAG}-conv run_convert.sbatch $R $OBJ $S)
  t=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=${S}_contact \
      APPTAINERENV_CURR=tracking_250m_abl_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=5:00:00 --dependency=afterok:$c --job-name=${TAG}-train run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward.sh)
  e=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_NAME=${S}_contact APPTAINERENV_CAMP=0.8,-1.1,0.8 \
      APPTAINERENV_CAMT=0,0,0 APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=2:00:00 --dependency=afterok:$t --job-name=${TAG}-eval run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_generic.sh)
  f=$(APPTAINERENV_NAME=${S}_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=1:00:00 --dependency=afterok:$e --job-name=${TAG}-evlast run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_last.sh)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $R $S $OBJ $c $t $e $f | tee -a $LOG
done; done
