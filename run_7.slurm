#!/bin/bash
#SBATCH --job-name=pymatching_L9_L15
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

module load python/3.10
source $PSCRATCH/envs/nde/bin/activate
cd $HOME/end_time

# PyMatching is CPU-based. Each (L, p) point uses 262144 shots with q=p.
experiments=(
    "--L 9 --seed 709"
    "--L 11 --seed 711"
    "--L 13 --seed 713"
    "--L 15 --seed 715"
)

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p "$dirname"
cd "$dirname"
export SLURM_CPU_BIND="cores"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

job_count=0
for params in "${experiments[@]}"; do
    echo "Starting PyMatching Experiment: $params"
    srun --ntasks=1 --mem=24G \
        python "$currdir/scripts/pymatching_threshold.py" $params \
        --p 0.010 0.0125 0.015 0.0175 0.020 0.025 0.030 0.035 0.040 0.045 \
        --shots 262144 --batch_size 2048 \
        --output "pymatching_exp_${job_count}.csv" \
        > "log_exp_${job_count}.txt" 2>&1 &
    sleep 2
    job_count=$((job_count + 1))
done
wait
