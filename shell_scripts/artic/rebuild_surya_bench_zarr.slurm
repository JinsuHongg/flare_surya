#!/usr/bin/env bash
#SBATCH --job-name=surya_bench_224_v2
#SBATCH --partition=qGPU24
#SBATCH --array=2010-2024%4
#SBATCH --output=surya_bench_224_v2_%A_%a.out
#SBATCH --error=surya_bench_224_v2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flare_surya

OUT=/scratch/users/jhong36/surya-bench-224-v2.zarr
TEMP_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
YEAR="${SLURM_ARRAY_TASK_ID:?This script must run as a year array}"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "year=$YEAR output_path=$OUT temporary_directory=$TEMP_DIR"

# The builder validates the output identity and resumes matching checkpoints.
# It writes annual Zarr datasets beneath OUT for the configured 2010-2024 range.
srun python scripts/preprocessing/rebuild_surya_bench_zarr.py \
  output_path="$OUT" \
  start_year=2010 \
  end_year=2024 \
  selected_year="$YEAR" \
  target_size=224 \
  temporary_directory="$TEMP_DIR" \
  dry_run=false
