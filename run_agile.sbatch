#!/bin/bash
#SBATCH --job-name=v2d-agile
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# AGILE (arXiv 2602.04672, minus mesh generation) for one episode.
# Needs: toolkit core run (recon_hawor/), Stage 5 objects (objects/), [Stage 3 stereo depth].
# Usage: sbatch run_agile.sbatch <run_name> [--depth moge|stereo] [run_agile.py args...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$1; shift
DEPTH=moge; prev=""
for a in "$@"; do [ "$prev" = "--depth" ] && DEPTH=$a; prev=$a; done

# Phase A (host env): interaction onset frame + FoundationPose registration at it.
if [ -z "${SKIP_PREPARE:-}" ]; then cd ~/v2d-vulcan/agile && python prepare.py "$NAME" --depth "$DEPTH"; fi

# Phase B (agile image): §3.2 init + §3.3 contact-aware optimisation.
TAG_FILE=$V2D_ROOT/runs/$NAME/agile_tag_${SLURM_JOB_ID}.txt
RUN_ARGS=(--rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code"
          -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" -e HF_HOME="$HF_HOME" -e AGILE_TAG_FILE="$TAG_FILE")
docker run "${RUN_ARGS[@]}" agile python /code/agile/run_agile.py "$NAME" "$@"

# 4D visualisation: camera view over the video + third-person view.
TAG=$(cat "$TAG_FILE")
docker run "${RUN_ARGS[@]}" agile python /code/agile/render_video.py "$NAME" --tag "$TAG"
