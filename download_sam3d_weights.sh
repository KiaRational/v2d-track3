#!/bin/bash
# Download the gated SAM 3D Objects weights (facebook/sam-3d-objects) on the LOGIN node (compute nodes have no
# route to huggingface.co). Plain curl + jq, no Python. Needs approved access on huggingface.co and HF_TOKEN
# exported in this shell. Resumable: re-run to continue; files already complete (size matches) are skipped.
# Usage:  read -rs HF_TOKEN && export HF_TOKEN && bash ~/v2d-vulcan/download_sam3d_weights.sh; unset HF_TOKEN
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN first (read -rs HF_TOKEN && export HF_TOKEN)}"
REPO=facebook/sam-3d-objects
S=/scratch/kghasemz/v2d/third_party/sam-3d-objects
DL=$S/checkpoints/hf-download
mkdir -p "$DL"
AUTH="Authorization: Bearer $HF_TOKEN"

LIST=$(curl -sfL -H "$AUTH" "https://huggingface.co/api/models/$REPO/tree/main?recursive=true") || {
  echo "cannot list $REPO: access not approved yet, or the token is wrong/expired" >&2; exit 1; }
echo "$LIST" | jq -r '.[] | select(.type=="file") | "\(.size)\t\(.path)"' > "$DL/.files.tsv"
echo "$(wc -l < "$DL/.files.tsv") files, $(jq '[.[] | select(.type=="file") | .size] | add' <<< "$LIST" | numfmt --to=iec) total"

while IFS=$'\t' read -r size path; do
  out="$DL/$path"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
    echo "ok    $path"; continue
  fi
  mkdir -p "$(dirname "$out")"
  echo "get   $path ($(numfmt --to=iec "$size"))"
  curl -fL --retry 5 --retry-delay 5 -C - -H "$AUTH" -o "$out" "https://huggingface.co/$REPO/resolve/main/$path"
  [ "$(stat -c %s "$out")" = "$size" ] || { echo "size mismatch for $path, re-run to resume" >&2; exit 1; }
done < "$DL/.files.tsv"

if [ -d "$DL/checkpoints" ]; then
  mkdir -p "$S/checkpoints/hf"
  cp -rl "$DL/checkpoints/." "$S/checkpoints/hf/" 2>/dev/null || cp -r "$DL/checkpoints/." "$S/checkpoints/hf/"
fi
echo "done -> $S/checkpoints/hf"
ls -la "$S/checkpoints/hf"
