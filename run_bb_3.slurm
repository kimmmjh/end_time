#!/bin/bash
#SBATCH --job-name=bbc_raw144_high
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

# High-p BB144 sweep. Together with run_bb_2 this spans p=0.001--0.010.
experiments=(
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.005 --measurement_error_rate=0.005 --bb_idle_error_rate=0 --seed=14401005"
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.006 --measurement_error_rate=0.006 --bb_idle_error_rate=0 --seed=14401006"
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.008 --measurement_error_rate=0.008 --bb_idle_error_rate=0 --seed=14401008"
    "$BB_CIRCUIT_NO_OSD_BB144_ARGS --p=0.010 --measurement_error_rate=0.010 --bb_idle_error_rate=0 --seed=14401010"
)

run_bb_experiments "${experiments[@]}"
