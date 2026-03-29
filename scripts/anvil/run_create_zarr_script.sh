#!/bin/bash -l
# FILENAME: run_create_zarr_script.sh

##SBATCH -A cis251356-ai
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=creat_zarr
#SBATCH --output=logs/%x_%j.out   # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err    # separate error log (optional)

# Clean environment
module purge
module --force purge
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

# Activate venv
source $SLURM_SUBMIT_DIR/.venv/bin/activate
cd flare_surya/utils

# Force libraries to use only 1 thread per worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1

unset SLURM_CPUS_PER_TASK

# Run Python directly (No srun needed for single node)
python create_surya_bench_zarr_multi_threads.py
