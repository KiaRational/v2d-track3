# Login-node driver (sbatch only): convert -> train (real-time playback) -> eval (every 20M ckpt + video) ->
# official eval of the last ckpt with the per-slot object trajectory export (submission input).
# = submit_all.sh (same recipe: tracking_250m_abl_contact, table + floor, depenetrate, rest settle) with
#   * every official object of the episode, in kit slot order (wooden_spoon included: reconstructed SAM 3D mesh);
#   * RT playback from the first run: sim.fps=<video fps, float32-exact> task.motion_speed=1.0 (1 row per frame).
# Usage: HANDS=mano_fit_hop_phys TAG=hopsp bash submit_all_rt.sh eval_list.txt
# Seq names: ${TAG}_e<NN> / ${TAG}_p<NN>; runs fc_out/<seq>_rt. Job ids -> $LOG.
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
HANDS=${HANDS:?}; TAG=${TAG:?}
FPS=${FPS:-20.254119873046875}  # float32(20.25411910142744), the fps of every episode
EXTRA="sim.fps=$FPS task.motion_speed=1.0"
KITJ=/scratch/kghasemz/v2d/submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json
LOG=/scratch/kghasemz/v2d/robot/submit_${TAG}.tsv
declare -A DEPS=()
if [ -n "${DEPS_FILE:-}" ]; then while read -r r j; do DEPS[$r]=$j; done < "$DEPS_FILE"; fi
for L in "$@"; do for R in $(cat "$L"); do
  ep=$((10#${R##*_ep})); NN=$(printf %02d $ep)
  if [[ $R == evaluation_* ]]; then
    S=${TAG}_e$NN; OBJ=$(jq -r --arg e $ep '.episodes[$e] | join(",")' $KITJ)
  else
    S=${TAG}_p$NN; OBJ=all
  fi
  N=${S}_rt
  DEPR=${DEPS[$R]:-}  # optional per-run upstream job (DEPS_FILE: "<run> <jobid>" lines)
  c=$(V2D_HANDS=$HANDS sbatch --parsable --exclude=rack02-06 ${DEPR:+--dependency=afterok:$DEPR} --job-name=${TAG}-conv run_convert.sbatch $R $OBJ $S)
  t=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=$N APPTAINERENV_EXTRA="$EXTRA" \
      APPTAINERENV_CURR=tracking_250m_abl_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=5:00:00 --exclude=rack02-06 --dependency=afterok:$c --job-name=${TAG}-train run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward.sh)
  e=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_NAME=$N APPTAINERENV_CAMP=0.8,-1.1,0.8 APPTAINERENV_CAMT=0,0,0 \
      APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=2:00:00 --exclude=rack02-06 --dependency=afterok:$t --job-name=${TAG}-eval run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_generic.sh)
  f=$(APPTAINERENV_NAME=$N APPTAINERENV_HUMAN_MOTION_DATA_DIR=/scratch/kghasemz/v2d/robot/hmd \
      sbatch --parsable --time=1:00:00 --exclude=rack02-06 --dependency=afterok:$e --job-name=${TAG}-evlast run_fc.sbatch \
      /home/kghasemz/v2d-vulcan/robot/cmds/eval_last.sh)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $R $N $OBJ $c $t $e $f | tee -a $LOG
done; done
