#!/bin/bash
#SBATCH --job-name=bbc_replicates
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

module load python/3.10 || exit 1
source "$PSCRATCH/envs/nde/bin/activate" || exit 1
ROOT="$HOME/end_time"
cd "$ROOT" || exit 1
source "$ROOT/scripts/run_bb_slurm_batch.sh" || exit 1
source "$ROOT/scripts/bb_neural_slurm_defaults.sh" || exit 1

# Two additional seeds at the central p=0.004 point. Combined with the seed-A
# points in run_bb_0/run_bb_2, this gives three training replicates per code.
experiments=(
    "$BB_CIRCUIT_BB72_ARGS --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=7202004"
    "$BB_CIRCUIT_BB72_ARGS --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=7203004"
    "$BB_CIRCUIT_BB144_ARGS --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=14402004"
    "$BB_CIRCUIT_BB144_ARGS --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=14403004"
)

run_bb_experiments "${experiments[@]}"
