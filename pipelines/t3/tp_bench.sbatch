#!/bin/bash
#SBATCH --job-name=tp-bench
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# DEV ONLY: public-20 mocap accuracy of c3 vs the new chain
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_recon V2D_DATA=/scratch/kghasemz/v2d/data/track_3
O=/scratch/kghasemz/v2d/beat1/diag/bench_tp.jsonl; : > $O
cat /scratch/kghasemz/v2d/beat1/pub20_c3.txt | xargs -P 20 -I{} python /scratch/kghasemz/v2d/beat1/bench_pose.py {} fused_bfix_recon tp_recon_vit tp_recon tpf_recon tp_hg_recon tphgf_recon >> $O
jq -s -c 'map(select(.moving!=null or .rest!=null)) | group_by(.variant)[] | {v: .[0].variant, n: length, mov_ok: ([.[].moving.ok15deg_3cm|select(.!=null)]|add/length), mov_rot: ([.[].moving.rot_med|select(.!=null)]|add/length), mov_tr: ([.[].moving.tr_med_cm|select(.!=null)]|add/length), rest_ok: ([.[].rest.ok15deg_3cm|select(.!=null)]|add/length), jumps: ([.[].jumps20]|add)}' $O
