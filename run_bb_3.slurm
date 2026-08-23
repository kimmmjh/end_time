#!/bin/bash
#SBATCH --job-name=bb_iteration_controls
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

# The archived reference is T=12. Every run also reports a same-shot vanilla
# BP4 result at its own T, so this supplies T=6/12/24 iteration controls.
ITERATION_MODEL_ARGS="--bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

experiments=(
    "$BB_NEURAL_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=6 --code=bb72 --p=0.08 --seed=7208"
    "$BB_NEURAL_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=24 --code=bb72 --p=0.08 --seed=7208"
    "$BB_NEURAL_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=6 --code=bb144 --p=0.08 --seed=14408"
    "$BB_NEURAL_TRAINING_ARGS $ITERATION_MODEL_ARGS --bp_iterations=24 --code=bb144 --p=0.08 --seed=14408"
)

run_bb_experiments "${experiments[@]}"
