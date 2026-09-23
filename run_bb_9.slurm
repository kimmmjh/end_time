#!/bin/bash
#SBATCH --job-name=bbcnn_d2_seeds
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

# Preview locally: BB_DRY_RUN=1 bash run_bb_9.slurm
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

# Two additional training/test seeds per code at p=0.06, depth=2.
# Depolarizing code capacity; each run reports CNN and CNN+direct OSD-0.
# 100 epochs, validation 4096 shots every 5 epochs, final test 65536 shots.
# Matrix, seeds and submission instructions: docs/bb_tanner_cnn.md
experiments=(
    "$(bb_tanner_cnn_experiment_args bb72 2 0.06 7202060)"
    "$(bb_tanner_cnn_experiment_args bb72 2 0.06 7203060)"
    "$(bb_tanner_cnn_experiment_args bb144 2 0.06 14402060)"
    "$(bb_tanner_cnn_experiment_args bb144 2 0.06 14403060)"
)

BB_EXPERIMENT_LABELS=(
    "bb72_tanner_cnn_depth2_p0.06_seed7202060"
    "bb72_tanner_cnn_depth2_p0.06_seed7203060"
    "bb144_tanner_cnn_depth2_p0.06_seed14402060"
    "bb144_tanner_cnn_depth2_p0.06_seed14403060"
)

run_bb_experiments "${experiments[@]}"
