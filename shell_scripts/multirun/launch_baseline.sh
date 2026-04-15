#!/usr/bin/bash

# Launch multiple training_baseline.py jobs with different configs
# Usage: ./launch_baseline.sh [config1 config2 ...]
# Example: ./launch_baseline.sh alexnet_c resnet18_c alexnet_m

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$#" -eq 0 ]; then
  CONFIGS=("alexnet_c" "alexnet_m" "resnet18_c" "resnet18_m")
else
  CONFIGS=("$@")
fi

echo "Launching ${#CONFIGS[@]} baseline training jobs..."
echo "Configs: ${CONFIGS[*]}"
echo ""

for config in "${CONFIGS[@]}"; do
  echo "Submitting job for config: $config"
  qsub -N "baseline_${config}" -v TASK_NAME="$config" "$SCRIPT_DIR/../nas/train_baseline_gh.pbs"
  sleep 1
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
