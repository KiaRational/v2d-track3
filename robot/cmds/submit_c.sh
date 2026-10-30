# Compliant recipe (no public data anywhere): every run's hoi4d_c reconstruction (our meshes_recon, fused_all_recon
# objects, hands_c hands; versions.json _current after `refine --version initial --name c1`) -> convert into its own
# motion-data tree robot/hmd_c (no provided collision meshes) -> physics assets robot/hmd_c_phys (v3 masses) ->
# train BOTH FlashSAC (train_v2d_reward_hmd.sh, tracking_250m_abl_contact) and PPO (train_ppo_v2d.sh,
# ppo_v2d_contact) -> evals. Usage: TAG=c1 bash submit_c.sh <runs.txt>. Seqs <TAG>_eNN / <TAG>_pNN; runs
# fc_out/<seq>_sac and fc_out/<seq>_ppo. Conversions one at a time (they rewrite the shared object meshes of the tree),
# assets once after the last, then the trainings.
set -euo pipefail
cd /home/kghasemz/v2d-vulcan/robot
HOLD=${HOLD:+--hold}   # HOLD=1: trainings queued held (scontrol release <ids> to start)
TAG=${TAG:-c1}; LOG=/scratch/kghasemz/v2d/beat1/${TAG}_runs.tsv; B1=/scratch/kghasemz/v2d/beat1
FPS=20.254119873046875; EXTRA="sim.fps=$FPS task.motion_speed=1.0"
HMDC=${HMDC:-/scratch/kghasemz/v2d/robot/hmd_c}; HMDP=${HMDP:-/scratch/kghasemz/v2d/robot/hmd_c_phys}; RGCC=${RGCC:-/scratch/kghasemz/v2d/robot/rg_cache_c}
KITJ=/scratch/kghasemz/v2d/submission_kit_new/unz/v2d_submission_kit/data/track_3_evaluation_objects.json
declare -A META
PREVC=; SEQS=()
for R in $(cat "$1"); do
  REG=/scratch/kghasemz/v2d/runs/$R/hoi4d_c/versions.json
  CUR=$(jq -r '._current // "initial"' $REG); OV=$(jq -r --arg v "$CUR" '.[$v].objects_variant' $REG); HV=$(jq -r --arg v "$CUR" '.[$v].hands_variant' $REG)
  ep=$((10#${R##*_ep})); NN=$(printf %02d $ep)
  if [[ $R == evaluation_* ]]; then S=${TAG}_e$NN; OBJ=$(jq -r --arg e $ep '.episodes[$e] | join(",")' $KITJ); else S=${TAG}_p$NN; OBJ=all; fi
  c=$(V2D_OBJ_DIR=objects_recon V2D_HANDS=$HV V2D_OBJECT_VARIANT=$OV V2D_MESH_DIR=meshes_recon V2D_PROVIDED_COLLISION=none V2D_HMD=$HMDC \
      V2D_RG_CACHE=$RGCC sbatch --parsable --exclude=rack02-06 ${PREVC:+--dependency=afterany:$PREVC} \
      --job-name=${TAG}-conv run_convert.sbatch $R $OBJ $S); PREVC=$c
  META[$R]="$S $CUR $OV $HV $c"; SEQS+=($S)
done
a=$(sbatch --parsable --exclude=rack02-06 --dependency=afterany:$PREVC $B1/phys_c.sbatch)
p=$(HMD_SRC=$HMDC HMD_DST=$HMDP sbatch --parsable --exclude=rack02-06 --dependency=afterok:$a --job-name=${TAG}-pq \
    --export=ALL,HMD_SRC=$HMDC,HMD_DST=$HMDP $B1/phys_pq.sbatch "${SEQS[@]}")
for R in $(cat "$1"); do
  read -r S CUR OV HV c <<< "${META[$R]}"
  ts=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=${S}_sac APPTAINERENV_HMD=$HMDP APPTAINERENV_EXTRA="$EXTRA" \
       APPTAINERENV_CURR=tracking_250m_abl_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDP \
       sbatch --parsable $HOLD --time=7:00:00 --exclude=rack02-06 --dependency=afterok:$p --job-name=${TAG}-sac run_fc.sbatch \
       /home/kghasemz/v2d-vulcan/robot/cmds/train_v2d_reward_hmd.sh)
  tp=$(FC_SRC=/scratch/kghasemz/v2d/fc_src_rewards APPTAINERENV_SEQ=$S APPTAINERENV_NAME=${S}_ppo APPTAINERENV_HMD=$HMDP APPTAINERENV_EXTRA="$EXTRA" \
       APPTAINERENV_CURR=ppo_v2d_contact APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDP \
       sbatch --parsable $HOLD --time=7:00:00 --exclude=rack02-06 --dependency=afterok:$p --job-name=${TAG}-ppo run_fc.sbatch \
       /home/kghasemz/v2d-vulcan/robot/cmds/train_ppo_v2d.sh)
  es=$(APPTAINERENV_NAME=${S}_sac APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDP \
       sbatch --parsable --time=1:00:00 --exclude=rack02-06 --dependency=afterany:$ts --job-name=${TAG}-evsac run_fc.sbatch \
       /home/kghasemz/v2d-vulcan/robot/cmds/eval_last.sh)
  ep_=$(APPTAINERENV_NAME=${S}_ppo APPTAINERENV_HUMAN_MOTION_DATA_DIR=$HMDP \
       sbatch --parsable --time=1:00:00 --exclude=rack02-06 --dependency=afterany:$tp --job-name=${TAG}-evppo run_fc.sbatch \
       $B1/eval_ppo.sh)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $R $S $CUR $OV $HV $c $ts $tp $es $ep_ | tee -a $LOG
done
