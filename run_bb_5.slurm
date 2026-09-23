#!/bin/bash
#SBATCH --job-name=bbcnn_bb72_d2
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Preview locally: BB_DRY_RUN=1 bash run_bb_5.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
cd "$ROOT"
source "$ROOT/scripts/run_bb_slurm_batch.sh"
source "$ROOT/scripts/bb_tanner_cnn_slurm_defaults.sh"
BB_CAMPAIGN="bb_tanner_cnn_2026_09"
BB_PYTHON_ENTRYPOINT="$ROOT/main.py"

# Default depth=2 CNN: check -> qubit -> check -> qubit.
# Depolarizing code capacity; each run reports CNN and CNN+direct OSD-0.
# 100 epochs, validation 4096 shots every 5 epochs, final test 65536 shots.
# Matrix, seeds and submission instructions: docs/bb_tanner_cnn.md
experiments=(
    "$(bb_tanner_cnn_experiment_args bb72 2 0.02 7201020)"
    "$(bb_tanner_cnn_experiment_args bb72 2 0.04 7201040)"
    "$(bb_tanner_cnn_experiment_args bb72 2 0.06 7201060)"
    "$(bb_tanner_cnn_experiment_args bb72 2 0.08 7201080)"
)

BB_EXPERIMENT_LABELS=(
    "bb72_tanner_cnn_depth2_p0.02_seed7201020"
    "bb72_tanner_cnn_depth2_p0.04_seed7201040"
    "bb72_tanner_cnn_depth2_p0.06_seed7201060"
    "bb72_tanner_cnn_depth2_p0.08_seed7201080"
)

run_bb_experiments "${experiments[@]}"
