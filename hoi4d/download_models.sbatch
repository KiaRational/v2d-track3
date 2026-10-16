#!/bin/bash
#SBATCH --job-name=hoi4d-dl
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Public (ungated) weights for hoi4d, plain curl (compute-node proxy allows huggingface.co + github.com).
#   Qwen/Qwen3-VL-32B-Instruct-FP8 -> weights/qwen3-vl-32b-fp8 ; CoTracker3 repo + scaled_offline.pth
set -euo pipefail
W=/scratch/kghasemz/v2d/weights
get_repo() {  # $1 = HF repo, $2 = dest
  mkdir -p "$2"
  curl -sfL "https://huggingface.co/api/models/$1/tree/main?recursive=true" \
    | jq -r '.[] | select(.type=="file") | "\(.size)\t\(.path)"' > "$2/.files.tsv"
  while IFS=$'\t' read -r size path; do
    out="$2/$path"
    if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then continue; fi
    mkdir -p "$(dirname "$out")"
    curl -fL --retry 5 --retry-delay 5 -C - -s -o "$out" "https://huggingface.co/$1/resolve/main/$path"
    [ "$(stat -c %s "$out")" = "$size" ] || { echo "size mismatch $path" >&2; exit 1; }
    echo "ok $path"
  done < "$2/.files.tsv"
}
get_repo Qwen/Qwen3-VL-32B-Instruct-FP8 $W/qwen3-vl-32b-fp8
mkdir -p $W/cotracker3
curl -fL --retry 5 -s -o $W/cotracker3/scaled_offline.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
curl -fL --retry 5 -s -o $W/cotracker3/scaled_online.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
T=/scratch/kghasemz/v2d/third_party
[ -d $T/co-tracker ] || git clone -q https://github.com/facebookresearch/co-tracker $T/co-tracker
git -C $T/co-tracker rev-parse HEAD
du -sh $W/qwen3-vl-32b-fp8 $W/cotracker3
