#!/bin/bash
# Aggregate eval_objects.json over all public runs: mean per variant over objects.
# Usage: bash summarize_eval.sh [variants...]
V=${*:-fp icp fused_free fused fused_sym agile agile_hc}
R=/scratch/kghasemz/v2d/runs
printf "%-11s %4s %8s %8s %8s %9s %s\n" variant nobj trans_cm ADDS_cm rot_deg jitter_mm "wins_vs_fp(trans)"
for v in $V; do
  jq -rs --arg v "$v" '
    [ .[] | .variants as $vs | ($vs.fp.objects // {}) as $fp | ($vs[$v].objects // {}) | to_entries[]
      | select(.value.trans_err_cm_mean != null and .value.ADDS_cm_mean != null and .value.rot_err_deg_mean != null and .value.jitter_mm != null)
      | {t: .value.trans_err_cm_mean, a: .value.ADDS_cm_mean, r: .value.rot_err_deg_mean, j: .value.jitter_mm,
         w: (if $fp[.key].trans_err_cm_mean != null and .value.trans_err_cm_mean < $fp[.key].trans_err_cm_mean then 1 else 0 end)} ]
    | if length == 0 then "\($v) 0" else
      "\($v) \(length) \((map(.t)|add/length*100|round/100)) \((map(.a)|add/length*100|round/100)) \((map(.r)|add/length*10|round/10)) \((map(.j)|add/length*100|round/100)) \(map(.w)|add)" end' \
    $(printf "$R/%s/${EVAL_FILE:-eval_objects.json} " $(cat /scratch/kghasemz/v2d/public_runs.txt)) | awk '{printf "%-11s %4s %8s %8s %8s %9s %s\n",$1,$2,$3,$4,$5,$6,$7}'
done
