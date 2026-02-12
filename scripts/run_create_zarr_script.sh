#!/bin/bash -l
# FILENAME: run_create_zarr_script.sh

##SBATCH -A cis251356-ai
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
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

# Activate your Python environment
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Move to your working directory
cd flare_surya/utils

# Run your training script
srun python create_surya_bench_zarr.py