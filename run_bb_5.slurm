#!/bin/bash
#SBATCH --job-name=bbc_global
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

# Global sharing uses one residual rule for every DEM edge. Orbit sharing remains
# the equivariant default; this tests whether distinguishing space-time edge
# types is actually useful. Baseline orbit runs use the same seeds.
GLOBAL_MODEL_ARGS="--bp_iterations=12 --bp_residual_hidden_dim=32 --bp_orbit_embedding_dim=8 --bp_parameter_sharing=global --bp_normalisation=0.625 --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

experiments=(
    "$BB_CIRCUIT_BB72_TRAINING_ARGS $GLOBAL_MODEL_ARGS --p=0.003 --measurement_error_rate=0.003 --bb_idle_error_rate=0 --seed=7201003"
    "$BB_CIRCUIT_BB72_TRAINING_ARGS $GLOBAL_MODEL_ARGS --p=0.005 --measurement_error_rate=0.005 --bb_idle_error_rate=0 --seed=7201005"
    "$BB_CIRCUIT_BB144_TRAINING_ARGS $GLOBAL_MODEL_ARGS --p=0.003 --measurement_error_rate=0.003 --bb_idle_error_rate=0 --seed=14401003"
    "$BB_CIRCUIT_BB144_TRAINING_ARGS $GLOBAL_MODEL_ARGS --p=0.005 --measurement_error_rate=0.005 --bb_idle_error_rate=0 --seed=14401005"
)

run_bb_experiments "${experiments[@]}"
