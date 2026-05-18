#!/bin/bash
#SBATCH --job-name=end_time_test
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=single:1

module load python/3.10
source $PSCRATCH/envs/nde/bin/activate
cd $HOME/end_time

experiments=(
    "--noise_model=phenomenological --measurement_error_rate=0.01 --p=0.01 --loss_fn=ce --epochs=300 --batch_size=128 --batches=512 --L=7 --channels 64 64 64 --depths 3 3 3 --amp_dtype=none"
)

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p $dirname
cd $dirname

job_count=0

for params in "${experiments[@]}"; do
    echo "Starting Experiment: $params"

    srun --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --mem=32G --cpu-bind=cores --gpu-bind=single:1 \
        python "$currdir/main.py" $params > log_exp_${job_count}.txt 2>&1 &

    sleep 3

    job_count=$((job_count + 1))
done

wait
