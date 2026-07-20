#!/bin/bash
#SBATCH --job-name=resume_gru_L11_p030
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch resume_L11_p030.slurm /absolute/path/to/model.pt" >&2
    exit 2
fi

checkpoint=$1
if [[ "$checkpoint" != /* ]]; then
    checkpoint="${SLURM_SUBMIT_DIR:-$PWD}/$checkpoint"
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
fi

module load python/3.10
source "$PSCRATCH/envs/nde/bin/activate"
cd "$HOME/end_time"

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p "$dirname"
cd "$dirname"
export SLURM_CPU_BIND="cores"

# --epochs=300 means 300 additional epochs. The checkpoint supplies epochs
# 1-300 and their loss/accuracy history, so the new plots span epochs 1-600.
srun --ntasks=1 --mem=16G \
    python "$currdir/main.py" \
    --architecture=convgru \
    --gru_channels=96 \
    --gru_layers=2 \
    --gru_kernel_size=3 \
    --noise_model=phenomenological \
    --measurement_error_rate=0.03 \
    --p=0.03 \
    --loss_fn=ce \
    --epochs=300 \
    --batch_size=64 \
    --batches=1024 \
    --eval_batches=512 \
    --L=11 \
    --channels 96 96 96 \
    --depths 4 4 4 \
    --lr=0.0003 \
    --amp_dtype=none \
    --save_model \
    --load_model "$checkpoint" \
    > log_resume.txt 2>&1
