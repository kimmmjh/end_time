#!/bin/bash
#SBATCH --job-name=bbc_relay_memory
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

# Preview locally: BB_DRY_RUN=1 bash run_bb_7.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
cd "$ROOT"
source "$ROOT/scripts/run_bb_slurm_batch.sh"
source "$ROOT/scripts/bb_neural_slurm_defaults.sh"
BB_CAMPAIGN="bb_neural_relay_2026_09"

# R=4,T=12,S=2: zero memory versus mean-matched constant later gamma=0.21.
# All runs: schema v2, legacy q=p, idle=0, 100 epochs, OSD disabled.
# Full design and comparison caveats: docs/bb_neural_relay_campaign.md
experiments=(
    "$(bb_relay_experiment_args bb72 zero_memory 0.004 7201004)"
    "$(bb_relay_experiment_args bb72 constant_memory 0.004 7201004)"
    "$(bb_relay_experiment_args bb144 zero_memory 0.004 14401004)"
    "$(bb_relay_experiment_args bb144 constant_memory 0.004 14401004)"
)

BB_EXPERIMENT_LABELS=(
    "bb72_zero_memory_p0.004_seed7201004"
    "bb72_constant_memory_p0.004_seed7201004"
    "bb144_zero_memory_p0.004_seed14401004"
    "bb144_constant_memory_p0.004_seed14401004"
)

run_bb_experiments "${experiments[@]}"
