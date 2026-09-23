#!/bin/bash
#SBATCH --job-name=bbcnn_bb144_d1
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

# Preview locally: BB_DRY_RUN=1 bash run_bb_8.slurm
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

# Local depth=1 control, matching depth=2 seeds, data streams and budgets.
# Depolarizing code capacity; each run reports CNN and CNN+direct OSD-0.
# 100 epochs, validation 4096 shots every 5 epochs, final test 65536 shots.
# Matrix, seeds and submission instructions: docs/bb_tanner_cnn.md
experiments=(
    "$(bb_tanner_cnn_experiment_args bb144 1 0.02 14401020)"
    "$(bb_tanner_cnn_experiment_args bb144 1 0.04 14401040)"
    "$(bb_tanner_cnn_experiment_args bb144 1 0.06 14401060)"
    "$(bb_tanner_cnn_experiment_args bb144 1 0.08 14401080)"
)

BB_EXPERIMENT_LABELS=(
    "bb144_tanner_cnn_depth1_p0.02_seed14401020"
    "bb144_tanner_cnn_depth1_p0.04_seed14401040"
    "bb144_tanner_cnn_depth1_p0.06_seed14401060"
    "bb144_tanner_cnn_depth1_p0.08_seed14401080"
)

run_bb_experiments "${experiments[@]}"
