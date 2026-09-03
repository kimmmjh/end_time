#!/bin/bash
#SBATCH --job-name=bbc_raw144_low
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

# OSD is fully disabled. Evaluation compares raw Neural BP2 with raw
# vanilla BP2, and checkpoints are selected by raw paired gain.

# Corrected circuit schema v2: twelve noisy cycles plus one closing frame.
# Use the same low-p grid as BB72 so distance scaling is directly comparable.
experiments=(
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.001 --measurement_error_rate=0.001 --bb_idle_error_rate=0 --seed=14401001"
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.002 --measurement_error_rate=0.002 --bb_idle_error_rate=0 --seed=14401002"
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.003 --measurement_error_rate=0.003 --bb_idle_error_rate=0 --seed=14401003"
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=14401004"
)

run_bb_experiments "${experiments[@]}"
