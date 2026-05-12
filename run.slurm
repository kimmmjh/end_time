#!/bin/bash
#SBATCH --job-name=end_time_test
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

module load python/3.10
source $PSCRATCH/envs/nde/bin/activate
cd $HOME/end_time

experiments=(
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --L=5 --channels 64 64 64 --depths 3 3 3"
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --L=7 --channels 64 64 64 --depths 3 3 3"
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --L=9 --channels 64 128 128 --depths 4 4 4"
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=400 --batch_size=32 --batches=2048 --L=11 --channels 64 128 128 64 --depths 4 4 4 4"
)

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p $dirname
cd $dirname

job_count=0

for params in "${experiments[@]}"; do
    echo "Starting Experiment: $params"

    srun -n 1 --exact -c 16 --mem=32G --gpus-per-task=1 --cpu-bind=cores python "$currdir/main.py" $params > log_exp_${job_count}.txt 2>&1 &

    sleep 3

    ((job_count++))
done
cd -

wait