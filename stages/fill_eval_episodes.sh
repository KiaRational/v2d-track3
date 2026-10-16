#!/bin/bash
# Fill the evaluation split's episode.json with the objects identified from video
# (<run>/objects_id.json, stages/objects_v2/identify_objects.py). The split ships no object
# list; every later stage reads episode.json. The original is kept as episode.orig.json.
# jq only: safe on the login node.
# Usage: bash fill_eval_episodes.sh [runs_list.txt]
set -euo pipefail
R=${V2D_ROOT:-/scratch/kghasemz/v2d}
LIST=${1:-$R/eval_runs.txt}
for r in $(cat "$LIST"); do
  d=$R/runs/$r
  [ -f "$d/objects_id.json" ] || { echo "$r: not identified yet"; continue; }
  [ -f "$d/episode.orig.json" ] || cp "$d/episode.json" "$d/episode.orig.json"
  jq --slurpfile id "$d/objects_id.json" --arg meshes "$R/meshes_obj" '
    ($id[0].objects) as $o
    | if ($o|length) == 0 then . + {objects: [], identification: "none"} else
      . + {objects: $o, primary_object: $o[0],
           object_prompt: ("a " + ($o[0] | gsub("_"; " "))),
           object_mesh: ($meshes + "/" + $o[0] + "/" + $o[0] + ".obj"),
           identification: "sam3_video"} end' "$d/episode.orig.json" > "$d/episode.json.tmp"
  mv "$d/episode.json.tmp" "$d/episode.json"
  echo "$r: $(jq -c .objects "$d/episode.json")"
done
