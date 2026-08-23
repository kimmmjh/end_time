#!/bin/bash
#SBATCH --job-name=bb_orbit_seed_c
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

# Orbit model replicate C.
experiments=(
    "$BB_NEURAL_COMMON_ARGS --code=bb72 --p=0.08 --seed=27208"
    "$BB_NEURAL_COMMON_ARGS --code=bb72 --p=0.10 --seed=27210"
    "$BB_NEURAL_COMMON_ARGS --code=bb144 --p=0.08 --seed=214408"
    "$BB_NEURAL_COMMON_ARGS --code=bb144 --p=0.10 --seed=214410"
)

run_bb_experiments "${experiments[@]}"
