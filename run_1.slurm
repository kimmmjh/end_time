#!/bin/bash
#SBATCH --job-name=end_L_size
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

experiments=(
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=256 --L=11 --channels 64 64 64 --depths 3 3 3 --lr=0.0003 --amp_dtype=none"
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=256 --L=11 --channels 64 128 128 --depths 3 3 3 --lr=0.0003 --amp_dtype=none"
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=200 --batch_size=32 --batches=1024 --eval_batches=512 --L=13 --channels 64 64 64 --depths 3 3 3 --lr=0.0003 --amp_dtype=none"
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=200 --batch_size=32 --batches=1024 --eval_batches=512 --L=13 --channels 64 96 96 --depths 3 3 3 --lr=0.0003 --amp_dtype=none"
)

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p $dirname
cd $dirname
export SLURM_CPU_BIND="cores"

job_count=0

for params in "${experiments[@]}"; do
    echo "Starting Experiment: $params"

    srun --ntasks=1 --mem=16G \
        python "$currdir/main.py" $params > log_exp_${job_count}.txt 2>&1 &

    sleep 3

    job_count=$((job_count + 1))
done

wait
