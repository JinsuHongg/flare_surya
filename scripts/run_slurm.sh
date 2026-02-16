#!/bin/bash -l
# FILENAME: run_slurm.sh

#SBATCH -A cis251356-ai
#SBATCH -p ai               # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --job-name=surya-flare
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

# # sanity check
# python - <<EOF
# import sys, numpy
# print("Python:", sys.version)
# print("NumPy:", numpy.__version__)
# print("NumPy path:", numpy.__file__)
# EOF

# Move to your working directory
cd flare_surya/task

# Run your training script
srun python finetuning.py
