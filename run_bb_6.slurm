#!/bin/bash
#SBATCH --job-name=bbc_raw_iterations
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

# T=12 is the sweep default. These same-seed T=6/T=24 runs test whether the
# detector-fault graph needs shorter or longer message propagation.
ITERATION_MODEL_ARGS="--bp_residual_hidden_dim=32 --bp_orbit_embedding_dim=8 --bp_parameter_sharing=orbit --bp_normalisation=0.625 --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

experiments=(
    "$BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=6 --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=7201004"
    "$BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=24 --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=7201004"
    "$BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=6 --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=14401004"
    "$BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=24 --p=0.004 --measurement_error_rate=0.004 --bb_idle_error_rate=0 --seed=14401004"
)

run_bb_experiments "${experiments[@]}"
