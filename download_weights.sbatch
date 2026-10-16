#!/bin/bash
#SBATCH --job-name=v2d-weights
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Fetch reconstruction weights through the compute-node proxy (HF, GitHub,
# Google Drive, NGC and download.pytorch.org are reachable). The HaMeR tarball
# (utexas.edu is blocked) is staged beforehand from the login node. Some
# toolkit downloaders `docker run --gpus` their image, hence the GPU request.
# Usage: sbatch download_weights.sbatch [--accept-nvidia-model-eula]
set -euo pipefail
source ~/v2d-vulcan/env.sh
W=/scratch/kghasemz/v2d/weights

TAR=$W/hamer/hamer_demo_data.tar.gz
if [ -f "$TAR" ] && [ ! -f "$W/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" ]; then
  echo "Extracting staged HaMeR data..."
  tar -xzf "$TAR" -C "$W/hamer" && rm -f "$TAR"
fi

# gdown follows redirects to doc-*.googleusercontent.com, which the proxy blocks
# (403); drive.usercontent.google.com works. Pre-fetch so the toolkit's
# downloaders see non-empty files and skip.
gdrive_fetch() {  # <file_id> <output_path>
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 -o "$2.part" "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"
  if head -c 512 "$2.part" | grep -qi '<html'; then echo "gdrive $1 returned HTML, not the file" >&2; rm -f "$2.part"; return 1; fi
  mv "$2.part" "$2"; echo "fetched $2 ($(du -h "$2" | cut -f1))"
}
DROID_ID=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh
gdrive_fetch $DROID_ID "$W/hawor/external/droid.pth"
gdrive_fetch 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr "$W/hawor/external/metric_depth_vit_large_800k.pth"
gdrive_fetch $DROID_ID "$W/droid_slam/droid.pth"

cd "$V2D_TOOLKIT/reconstruction"
./scripts/download_ego_reconstruction_weights.sh --mode hawor_mesh "$@"

# FoundationStereo isn't part of the toolkit's ego pipeline; we add it for metric
# depth from the ego stereo pair.
pip install -q --no-index --no-deps -e modules/v2d_foundation_stereo/docker
python -m v2d.foundation_stereo.docker.run_download_weights --output_dir data/weights/foundation_stereo

# MANO (license-gated, provided by the user): the toolkit's canonical location
# (docs/ego_e2e_setup.md) and every other location in HaWoR/HaMeR/WiLoR lookups.
for d in hand hand/models hamer/_DATA/data/mano hamer/_DATA/data/models wilor/pretrained_models wilor/pretrained_models/models; do
  mkdir -p "$W/$d"; for f in "$W"/mano/MANO_*.pkl; do [ -e "$W/$d/$(basename "$f")" ] || cp "$f" "$W/$d/"; done
done
du -sh "$W"/* 2>/dev/null
