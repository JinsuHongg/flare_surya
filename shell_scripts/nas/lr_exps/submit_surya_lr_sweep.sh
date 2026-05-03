#!/usr/bin/bash
# Submit multiple jobs sweeping learning rates for m_exp.yaml

# Set the task name based on the config file
TASK_NAME="m_exp"

# Define the learning rates to sweep
LEARNING_RATES=(1e-6 5e-6 1e-5 5e-5 1e-4)
WEIGHT_DECAYS=(0.05)

for lr in "${LEARNING_RATES[@]}"; do
  for wd in "${WEIGHT_DECAYS[@]}"; do
    echo "Submitting job for lr=${lr} wd=${wd}"

    # We pass the LR and WD as variables that the main script can use if updated
    # OR we override it directly in the command.
    # Given the structure, overriding directly in the command is safer.

    qsub -N "${TASK_NAME}_lr${lr}_wd${wd}" \
      -v TASK_NAME="${TASK_NAME}",LEARNING_RATE="${lr}",WEIGHT_DECAY="${wd}" \
      ./shell_scripts/nas/lr_exps/sweep_wrapper.pbs
  done
done
