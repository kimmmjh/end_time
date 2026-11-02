#!/bin/bash
#SBATCH --job-name=bbdem_continue
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=0-5
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
# See docs/bb_dem_followup_2026_11_02.md for selection, worker mapping and recovery.
# Preview: BB_DRY_RUN=1 bash run_bb_dem_continue.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
exec bash "$ROOT/scripts/run_bb_dem_followup.sh" train "${BASH_SOURCE[0]}" "$ROOT"
