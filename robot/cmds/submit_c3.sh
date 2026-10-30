# c3 compliant recipe: objects = fused_bfix_recon (fusion + untouched-static + flip-branch repair; no hoi4d refine,
# which added spin to resting pots), hands = hands_c, our meshes. Tree chain: convert (serial) -> robot/hmd_c3 ->
# physics assets + parquets -> robot/hmd_c3_phys -> walls-only fine collision (cups/pot openings stay open) ->
# robot/hmd_c3_fine -> FlashSAC, one training per recipe in $RECIPES (default: contact + task completion) -> eval ->
# rollout videos (viz/robot/<seq>_<recipe>.mp4). PPO dropped (c2: AUC 0.37 vs SAC 0.63).
# Usage: [HOLD=1] bash submit_c3.sh <runs.txt>
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
R=/scratch/kghasemz/v2d; B1=$R/beat1; TAG=${TAG:-c3}; LOG=$B1/${TAG}_runs.tsv; : > $LOG
HOLD=${HOLD:+--hold}
OV=${OV:-fused_bfix_recon}; HV=${HV:-hands_c}
HMDC=$R/robot/hmd_$TAG HMDP=$R/robot/hmd_${TAG}_phys HMDF=$R/robot/hmd_${TAG}_fine RGCC=$R/robot/rg_cache_$TAG
FPS=20.254119873046875; EXTRA="sim.fps=$FPS task.motion_speed=1.0"
KITJ=$R/submission_kit_new/unz/v2d_submission_kit/data/track_3_evaluation_objects.json
RECIPES=${RECIPES:-"contact:tracking_250m_abl_contact task:tracking_250m_task"}
PREVC=; SEQS=(); declare -A SQ
for RUN in $(cat "$1"); do
  # export requires the hands report to name the paired objects variant
  hr=$R/runs/$RUN/stereo/$HV/report.json; jq --arg v $OV '.args.variant = $v' $hr > $hr.tmp && mv $hr.tmp $hr
  ep=$((10#${RUN##*_ep})); NN=$(printf %02d $ep)
  if [[ $RUN == evaluation_* ]]; then S=${TAG}_e$NN; OBJ=$(jq -r --arg e $ep '.episodes[$e] | join(",")' $KITJ); else S=${TAG}_p$NN; OBJ=all; fi
  c=$(V2D_OBJ_DIR=objects_recon V2D_HANDS=$HV V2D_OBJECT_VARIANT=$OV V2D_MESH_DIR=meshes_recon V2D_PROVIDED_COLLISION=none \
      V2D_HMD=$HMDC V2D_RG_CACHE=$RGCC sbatch --parsable --exclude=rack02-06 ${PREVC:+--dependency=afterany:$PREVC} \
      --job-name=${TAG}-conv run_convert.sbatch $RUN $OBJ $S); PREVC=$c
  SEQS+=($S); SQ[$RUN]="$S $c"
done
a=$(HMDC=$HMDC HMDP=$HMDP sbatch --parsable --exclude=rack02-06 --dependency=afterany:$PREVC --job-name=${TAG}-phys $B1/phys_c.sbatch)
p=$(sbatch --parsable --exclude=rack02-06 --dependency=afterok:$a --job-name=${TAG}-pq --export=ALL,HMD_SRC=$HMDC,HMD_DST=$HMDP \
    $B1/phys_pq.sbatch "${SEQS[@]}")
f=$(APPTAINERENV_SRC=$HMDP APPTAINERENV_DST=$HMDF APPTAINERENV_SEQS=$(IFS=,; echo "${SEQS[*]}") sbatch --parsable --time=3:00:00 \
    --exclude=rack02-06 --dependency=afterok:$p --job-name=${TAG}-fine run_fc.sbatch $B1/fine_tree.sh)
VL=$B1/${TAG}_vid.txt; : > $VL; TR=()
for RUN in $(cat "$1"); do
  read -r S c <<< "${SQ[$RUN]}"; row="$RUN $S $c"
  for rc in $RECIPES; do
    nm=${rc%%:*}; cu=${rc##*:}
    t=$(FC_SRC=$R/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=${S}_$nm APPTAINERENV_HMD=$HMDF APPTAINERENV_EXTRA="$EXTRA" \
        APPTAINERENV_CURR=$cu APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDF sbatch --parsable $HOLD --time=7:00:00 --exclude=rack02-06 \
        --dependency=afterok:$f --job-name=${TAG}-sac run_fc.sbatch $HOME/v2d-vulcan/robot/cmds/train_v2d_reward_hmd.sh)
    e=$(APPTAINERENV_NAME=${S}_$nm APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDF sbatch --parsable --time=1:00:00 --exclude=rack02-06 \
        --dependency=afterany:$t --job-name=${TAG}-ev run_fc.sbatch $HOME/v2d-vulcan/robot/cmds/eval_last.sh)
    echo "${S}_$nm $HMDF" >> $VL; TR+=($t); row="$row $nm:$t:$e"
  done
  echo "$row" | tee -a $LOG
done
v=$(APPTAINERENV_LIST=$VL sbatch --parsable --time=6:00:00 --exclude=rack02-06 --dependency=afterany:$(IFS=:; echo "${TR[*]}") \
    --job-name=${TAG}-robotvid run_fc.sbatch $HOME/v2d-vulcan/robot/cmds/video_last_list.sh)
echo "phys $a pq $p fine $f videos $v" | tee -a $LOG
