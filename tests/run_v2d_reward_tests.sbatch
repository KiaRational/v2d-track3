#!/bin/bash
#SBATCH --job-name=v2d-reward-tests
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/kghasemz/v2d/code_tests/v2d_rewards/tests_%j.out
# CPU tests of the V2D reward terms (v2d-rewards worktree) + canonical-code acceptance of the stripped config.
module load apptainer/1.4.5
W=${FC_SRC:-/scratch/kghasemz/v2d/fc_src_rewards}/robotic_grounding/flash_chord
C=/scratch/kghasemz/v2d/fc_src/robotic_grounding/flash_chord
T=/scratch/kghasemz/v2d/code_tests/v2d_rewards; mkdir -p $T
P=/scratch/kghasemz/v2d/robot/hmd/ego_recon/processed/sequence_id=ep12_bluecup/robot_name=sharpa_wave
run() { apptainer exec --cleanenv --env PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=2,WARP_CACHE_PATH=$T/warp,V2D_STRIPPED_CONFIG_OUT=$T/stripped_config.json \
  -B /scratch -B /home/kghasemz/v2d-vulcan:/code:ro "$@"; }
echo "== unit tests (worktree)"
run -B $W:/workspace --pwd /workspace /scratch/kghasemz/v2d/sif/flash_chord.sif /opt/venv/bin/python -m pytest -q -p no:cacheprovider tests/objectives/test_v2d_reference.py tests/objectives/test_v2d_config.py 2>&1 | tail -15
echo "== canonical code accepts the stripped v2d config"
run -B $C:/workspace --pwd /workspace /scratch/kghasemz/v2d/sif/flash_chord.sif /opt/venv/bin/python /code/tests/v2d_rewards/check_canonical_accepts.py $T/stripped_config.json 2>&1 | tail -5
echo "== contact-location targets on ep12_bluecup"
run -B $W:/workspace --pwd /workspace /scratch/kghasemz/v2d/sif/flash_chord.sif /opt/venv/bin/python /code/tests/v2d_rewards/analyze_targets.py $P 2>&1 | tail -30
