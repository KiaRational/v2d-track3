#!/bin/bash
# Evaluation split episode.json from the OFFICIAL object lists and slot order
# (submission kit data/track_3_evaluation_objects.json). Only the 20 scored episodes are kept.
# The toolkit reconstruction needs one prompt+mesh: slot order is kept in "objects", and
# primary_object is the first slot other than wooden_spoon (reconstructed SAM 3D mesh), else wooden_spoon.
# Original kept as episode.orig.json. jq only.
set -euo pipefail
R=${V2D_ROOT:-/scratch/kghasemz/v2d}
OFF=$R/submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json
: > $R/eval_scored_runs.txt; : > $R/eval_spoon_only.txt
for ep in $(jq -r '.episodes | keys[]' $OFF | sort -n); do
  r=$(printf "evaluation_ep%06d" $ep); d=$R/runs/$r
  [ -d "$d" ] || { echo "$r: no run dir"; continue; }
  [ -f "$d/episode.orig.json" ] || cp "$d/episode.json" "$d/episode.orig.json"
  objs=$(jq -c --arg e "$ep" '.episodes[$e]' $OFF)
  prim=$(jq -r '([.[] | select(. != "wooden_spoon")][0] // .[0]) // empty' <<<"$objs")
  jq --argjson o "$objs" --arg p "$prim" --arg m "$R/meshes_obj" '
    . + {objects: $o, object_slots: $o, identification: "official_submission_kit"}
    + (if $p == "" then {primary_object: null} else
       {primary_object: $p, object_prompt: ("a " + ($p|gsub("_";" "))),
        object_mesh: ($m + "/" + $p + "/" + $p + ".obj")} end)' "$d/episode.orig.json" > "$d/episode.json"
  if [ -z "$prim" ]; then echo $r >> $R/eval_spoon_only.txt; else echo $r >> $R/eval_scored_runs.txt; fi
  echo "$r: $objs  primary=${prim:-NONE}"
done
