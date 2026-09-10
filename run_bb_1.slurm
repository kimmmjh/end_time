#!/bin/bash
#SBATCH --job-name=bbc_relay72_high
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

# Preview locally: BB_DRY_RUN=1 bash run_bb_1.slurm
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

# Relay reference, BB72 high-p: overlaps the completed OSD reference.
# All runs: schema v2, legacy q=p, idle=0, 100 epochs, OSD disabled.
# Full design and comparison caveats: docs/bb_neural_relay_campaign.md
experiments=(
    "$(bb_relay_experiment_args bb72 relay 0.005 7201005)"
    "$(bb_relay_experiment_args bb72 relay 0.006 7201006)"
    "$(bb_relay_experiment_args bb72 relay 0.008 7201008)"
    "$(bb_relay_experiment_args bb72 relay 0.010 7201010)"
)

BB_EXPERIMENT_LABELS=(
    "bb72_relay_p0.005_seed7201005"
    "bb72_relay_p0.006_seed7201006"
    "bb72_relay_p0.008_seed7201008"
    "bb72_relay_p0.010_seed7201010"
)

run_bb_experiments "${experiments[@]}"
