#!/usr/bin/bash

# Launch multiple finetuning_suryaflux.py jobs with different configs
# Usage: ./launch_suryaflux.sh [config1 config2 ...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default configs if none provided
if [ "$#" -eq 0 ]; then
  CONFIGS=("m_concat" "m_cross" "m_gated")
else
  CONFIGS=("$@")
fi

echo "Launching ${#CONFIGS[@]} SuryaFlux finetuning jobs..."
echo "Configs: ${CONFIGS[*]}"
echo ""

for config in "${CONFIGS[@]}"; do
  echo "Submitting job for config: $config"
  qsub -N "${config}" -v TASK_NAME="$config" "$SCRIPT_DIR/../nas/finetune_suryaflux_gh.pbs"
  sleep 1
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
