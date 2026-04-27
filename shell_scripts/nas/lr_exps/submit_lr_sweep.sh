#!/usr/bin/bash
# Submit multiple jobs sweeping learning rates for m_exp.yaml

# Set the task name based on the config file
TASK_NAME="m_exp"

# Define the learning rates to sweep
LEARNING_RATES=(1e-6 5e-6 1e-5 5e-5 1e-4)

for lr in "${LEARNING_RATES[@]}"; do
  echo "Submitting job for lr=${lr}"

  # We pass the LR as a variable that the main script can use if updated
  # OR we override it directly in the command.
  # Given the structure, overriding directly in the command is safer.

  # We need a small modification to the original script or command
  # to support the lr override.

  # Since I cannot modify the training script, I'll assume it accepts hydra overrides.

  qsub -N "${TASK_NAME}_lr${lr}" \
    -v TASK_NAME="${TASK_NAME}",LEARNING_RATE="${lr}" \
    ./shell_scripts/nas/lr_exps/sweep_wrapper.pbs
done
