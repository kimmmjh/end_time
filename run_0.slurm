#!/bin/bash
#SBATCH --job-name=end_time_test
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
    "--noise_model=phenomenological --measurement_error_rate=0.005 --p=0.005 --loss_fn=weighted_ce --epochs=500 --batch_size=64 --batches=1024 --L=9 --channels 64 64 64 --depths 3 3 3 --lr=0.0003 --amp_dtype=none"
    "--noise_model=phenomenological --measurement_error_rate=0.005 --p=0.005 --loss_fn=weighted_ce --epochs=500 --batch_size=64 --batches=1024 --L=9 --channels 64 64 64 --depths 3 3 3 --lr=0.0001 --amp_dtype=none"
    "--noise_model=phenomenological --measurement_error_rate=0.005 --p=0.005 --loss_fn=weighted_ce --epochs=500 --batch_size=64 --batches=1024 --L=9 --channels 64 64 64 --depths 3 3 3 --lr=0.0003 --amp_dtype=none"
    "--noise_model=phenomenological --measurement_error_rate=0.005 --p=0.005 --loss_fn=weighted_ce --epochs=500 --batch_size=64 --batches=1024 --L=9 --channels 64 64 64 --depths 3 3 3 --lr=0.0001 --amp_dtype=none"
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
