#!/bin/bash
#SBATCH --job-name=threshold_large_p030
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

module load python/3.10
source $PSCRATCH/envs/nde/bin/activate
cd $HOME/end_time

# The p=0.03 points for all four code distances.
experiments=(
    "--noise_model=phenomenological --measurement_error_rate=0.03 --p=0.03 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=256 --L=9 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.03 --p=0.03 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=256 --L=11 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.03 --p=0.03 --loss_fn=ce --epochs=300 --batch_size=32 --batches=1024 --eval_batches=512 --L=13 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.03 --p=0.03 --loss_fn=ce --epochs=300 --batch_size=16 --batches=2048 --eval_batches=1024 --L=15 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --save_model"
)

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p "$dirname"
cd "$dirname"
export SLURM_CPU_BIND="cores"

job_count=0
for params in "${experiments[@]}"; do
    echo "Starting Experiment $job_count: $params"
    srun --ntasks=1 --mem=16G \
        python "$currdir/main.py" $params \
        > "log_exp_${job_count}.txt" 2>&1 &
    sleep 3
    job_count=$((job_count + 1))
done
wait
