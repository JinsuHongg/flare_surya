#!/usr/bin/bash

# Launch multiple finetuning_suryaflux.py jobs with different configs
# Usage: ./launch_suryaflux.sh [config1 config2 ...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONFIGS=("${@:-m_suryaflux}")

echo "Launching ${#CONFIGS[@]} SuryaFlux finetuning jobs..."
echo "Configs: ${CONFIGS[*]}"
echo ""

for config in "${CONFIGS[@]}"; do
  echo "Submitting job for config: $config"
  qsub -N "suryaflux_${config}" -v TASK_NAME="$config" "$SCRIPT_DIR/../nas/finetune_suryaflux.pbs"
  sleep 1
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
