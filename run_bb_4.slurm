#!/bin/bash
#SBATCH --job-name=bbc_relay_seeds
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

# Preview locally: BB_DRY_RUN=1 bash run_bb_4.slurm
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

# Two extra p=0.004 seeds per code; three seeds including jobs 0/2.
# All runs: schema v2, legacy q=p, idle=0, 100 epochs, OSD disabled.
# Full design and comparison caveats: docs/bb_neural_relay_campaign.md
experiments=(
    "$(bb_relay_experiment_args bb72 relay 0.004 7202004)"
    "$(bb_relay_experiment_args bb72 relay 0.004 7203004)"
    "$(bb_relay_experiment_args bb144 relay 0.004 14402004)"
    "$(bb_relay_experiment_args bb144 relay 0.004 14403004)"
)

BB_EXPERIMENT_LABELS=(
    "bb72_relay_p0.004_seed7202004"
    "bb72_relay_p0.004_seed7203004"
    "bb144_relay_p0.004_seed14402004"
    "bb144_relay_p0.004_seed14403004"
)

run_bb_experiments "${experiments[@]}"
