#!/bin/bash
# Aggregate eval_global.json (eval_global.py) over public runs (login-safe, jq only).
# Usage: bash summarize_global.sh [-e] [variants...]    -e: also per-episode/object rows
#   Only objects present in ALL listed variants are averaged (fair comparison), per run.
PER=0; [ "${1:-}" = "-e" ] && { PER=1; shift; }
V=${*:-fp icp fused stereo_global_raw stereo_global sg_loc_raw sg_loc_anchor sg_top1 sg_oracle}
R=/scratch/kghasemz/v2d/runs
VJ=$(printf '%s\n' $V | jq -R . | jq -sc .)
jq -rs --argjson vs "$VJ" --argjson per $PER '
  [ .[] | .episode as $ep | .variants as $V
    | ([ $vs[] | ($V[.] // {}) | keys ] | if length == 0 then [] else reduce .[1:][] as $k (.[0]; . - (. - $k)) end) as $common
    | $common[] as $o | $vs[] as $v | ($V[$v][$o]) as $m
    | {ep: $ep, o: $o, v: $v, c: $m.centre_cm, t: $m.trans_cm, a: $m.ADDS_cm, r: $m.rot_deg, rs: $m.rot_sym_deg, j: $m.jitter_mm} ] as $rows
  | def f(x): if x == null then "   -" else (x*100|round/100|tostring) end;
    def mean(k): (map(k) | map(select(. != null))) as $x | if ($x|length)==0 then null else ($x|add/length) end;
  (if $per == 1 then ($rows | map("\(.ep|sub("public_ep0000";"ep")) \(.o) \(.v) \(f(.c)) \(f(.t)) \(f(.a)) \(f(.r)) \(f(.rs)) \(f(.j))") | .[]) else empty end),
  "variant nobj centre_cm trans_cm ADDS_cm rot_deg rot_sym_deg jitter_mm",
  ($vs[] as $v | $rows | map(select(.v == $v)) | "\($v) \(length) \(f(mean(.c))) \(f(mean(.t))) \(f(mean(.a))) \(f(mean(.r))) \(f(mean(.rs))) \(f(mean(.j)))")
' $R/public_*/eval_global.json | column -t
