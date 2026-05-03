#!/usr/bin/bash
# Submit multiple jobs sweeping learning rates for baseline training

TASK_NAME="alexnet_m"
LEARNING_RATES=(1e-6 5e-6 1e-5 5e-5 1e-4)
WEIGHT_DECAYS=(1e-3)

for lr in "${LEARNING_RATES[@]}"; do
  for wd in "${WEIGHT_DECAYS[@]}"; do
    echo "Submitting: $TASK_NAME with lr=$lr wd=$wd"
    qsub -N "${TASK_NAME}_lr${lr}_wd${wd}" \
      -v TASK_NAME="${TASK_NAME}",LEARNING_RATE="${lr}",WEIGHT_DECAY="${wd}" \
      ./shell_scripts/nas/lr_exps/train_lr_sweep.pbs
  done
done
