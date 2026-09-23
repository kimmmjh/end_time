#!/bin/bash
#SBATCH --job-name=bbc_bp72_high
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
BB_CAMPAIGN="bb_plain_bp_2026_09"
BB_PYTHON_ENTRYPOINT="$ROOT/scripts/evaluate_bb_circuit_bp.py"

# Ordinary BP only: first-valid stopping, paired caps 12/1000, 4096 shots.
# Schema v2, legacy q=p, idle=0. No training, Relay, or OSD.
# Experiment matrix and outputs: docs/bb_plain_bp_campaign.md
experiments=(
    "$(bb_plain_bp_experiment_args bb72 0.005 7201005 exp_0)"
    "$(bb_plain_bp_experiment_args bb72 0.006 7201006 exp_1)"
    "$(bb_plain_bp_experiment_args bb72 0.008 7201008 exp_2)"
    "$(bb_plain_bp_experiment_args bb72 0.010 7201010 exp_3)"
)

BB_EXPERIMENT_LABELS=(
    "bb72_plain_bp_p0.005_seed7201005"
    "bb72_plain_bp_p0.006_seed7201006"
    "bb72_plain_bp_p0.008_seed7201008"
    "bb72_plain_bp_p0.010_seed7201010"
)

run_bb_experiments "${experiments[@]}"
