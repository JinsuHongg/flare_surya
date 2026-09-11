# Current experiment plan

## Paper-facing matrix

The completed 24-hour forecasting-window records form the historical main
experiment story. The active 2-hour family is reported separately and is not
finalized merely because configurations or partial result tables exist.

| Story | Target | Training strategy | Models | Sampling intervals |
| --- | --- | --- | --- | --- |
| 24 h historical | C>=, M>= | Full/non-undersampled | Surya, AlexNet | 24 h, 12 h, 8 h |
| 24 h historical | M>= | Undersampled | Surya, AlexNet | 4 h, 3 h, 2 h |
| 24 h historical | X>= | Undersampled | Surya, AlexNet | 4 h, 3 h, 2 h, 1 h |
| 2 h active | C>= | Full/non-undersampled | Surya, AlexNet, ResNet18 | 24 h, 12 h, 8 h |
| 2 h active | M>= | Undersampled | Surya, AlexNet, ResNet18 | 2 h, 1 h |

The 2-hour C>= 24-hour-interval ResNet18 cell is not evaluated in the local
record. The 2-hour X>= configuration family exists, but its sampling-interval
provenance requires manual confirmation before it is added to a final matrix.

## Checkpoint and test procedure

For each model and sampling-interval configuration, select the validation
checkpoint with highest CSS. Then compare those retained checkpoints across
sampling intervals. Run Solar Cycle 25 testing only with a verified explicit
checkpoint and test index. Do not choose a final sampling interval from
incomplete 2-hour experiments.

See [24-hour experiments](../experiments/24h_forecasting.md),
[2-hour experiments](../experiments/2h_forecasting.md), and the
[evaluation plan](../experiments/evaluation_plan.md).
