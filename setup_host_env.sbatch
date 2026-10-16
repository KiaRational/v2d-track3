#!/bin/bash
#SBATCH --job-name=v2d-host-env
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Host-side venv for the toolkit orchestrator + visualization (Alliance wheelhouse).
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 opencv/4.11.0 arrow/25.0.0 ffmpeg/7.1.1

VENV=/scratch/kghasemz/v2d/venv-host
rm -rf "$VENV"
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scipy pillow opencv_python trimesh openexr pyarrow pandas \
    matplotlib imageio imageio_ffmpeg tqdm

M=/scratch/kghasemz/v2d/toolkit/reconstruction/modules
# Dependencies are satisfied above; --no-deps keeps pip away from PyPI builds.
pip install --no-deps -e $M/v2d_common -e $M/v2d_docker -e $M/v2d_depth \
  -e $M/v2d_anycalib/docker -e $M/v2d_geocalib/docker -e $M/v2d_droid_slam/docker \
  -e $M/v2d_ego_hand_reconstruction/docker -e $M/v2d_foundation_pose/docker \
  -e $M/v2d_grounding_dino/docker -e $M/v2d_gsplat_refinement/docker \
  -e $M/v2d_hamer/docker -e $M/v2d_hand_alignment/docker -e $M/v2d_hawor/docker \
  -e $M/v2d_mediapipe/docker -e $M/v2d_moge/docker -e $M/v2d_sam2/docker \
  -e $M/v2d_sam3d/docker -e $M/v2d_wilor/docker -e $M/v2d_pipelines

python - <<'EOF'
import cv2, pyarrow, pandas, trimesh, numpy, scipy, matplotlib, imageio, OpenEXR
import v2d.docker, v2d.pipelines
print("host env OK: cv2", cv2.__version__, "| numpy", numpy.__version__, "| pyarrow", pyarrow.__version__)
EOF
