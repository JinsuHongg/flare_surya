#!/usr/bin/bash

# Launch multiple finetuning.py jobs with different configs
# Usage: ./launch_finetune.sh [config1 config2 ...]
# Example: ./launch_finetune.sh c_focal x_focal m_focal

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default configs if none provided
if [ "$#" -eq 0 ]; then
	CONFIGS=("c_focal" "x_focal" "m_focal")
else
	CONFIGS=("$@")
fi

echo "Launching ${#CONFIGS[@]} finetuning jobs..."
echo "Configs: ${CONFIGS[*]}"
echo ""

for config in "${CONFIGS[@]}"; do
	echo "Submitting job for config: $config"
	qsub -N "finetune_${config}" -v TASK_NAME="$config" "$SCRIPT_DIR/../nas/finetune_flaresurya_gh.pbs"
	sleep 1
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
