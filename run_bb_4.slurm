#!/bin/bash
#SBATCH --job-name=bb_independent_xz
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

# Match the depolarizing X/Z component marginals with px=pz=2p/3 while
# removing their same-qubit correlation. Here --p is a reference label.
INDEPENDENT_ARGS="$BB_NEURAL_TRAINING_CORE_ARGS $BB_NEURAL_DEFAULT_MODEL_ARGS --bb_channel=independent_xz"

experiments=(
    "$INDEPENDENT_ARGS --code=bb72 --p=0.08 --x_error_rate=0.0533333333 --z_error_rate=0.0533333333 --seed=7208"
    "$INDEPENDENT_ARGS --code=bb72 --p=0.10 --x_error_rate=0.0666666667 --z_error_rate=0.0666666667 --seed=7210"
    "$INDEPENDENT_ARGS --code=bb144 --p=0.08 --x_error_rate=0.0533333333 --z_error_rate=0.0533333333 --seed=14408"
    "$INDEPENDENT_ARGS --code=bb144 --p=0.10 --x_error_rate=0.0666666667 --z_error_rate=0.0666666667 --seed=14410"
)

run_bb_experiments "${experiments[@]}"
