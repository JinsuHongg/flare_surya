#!/bin/bash -l
# FILENAME: run_slurm.sh

#SBATCH -A cis251356-ai
#SBATCH -p ai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --job-name=surya-flare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Clean environment
module purge
module load modtree/gpu      # Recommended for GPU workloads on Anvil
module load cuda/12.8        # Ensure this matches your .venv requirements

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

# Activate your Python environment
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Move to your working directory
cd flare_surya/task

# CRITICAL: PyTorch Lightning + NCCL environment variables
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1  # Better error messages

# --- Networking & Distributed Config ---
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 25000-30000 -n 1) # Dynamic port to avoid collisions

# NCCL Configuration for Ethernet (Anvil ai nodes)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1         # Forced Ethernet as per your requirement
export NCCL_SOCKET_IFNAME=^lo,docker
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run training
srun --cpu-bind=cores python finetuning.py --config-name="$1"