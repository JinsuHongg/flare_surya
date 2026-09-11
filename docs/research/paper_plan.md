# Paper plan

## Central question

When does Surya improve highly imbalanced solar flare forecasting relative to
AlexNet and ResNet18, and when is its advantage constrained by rare-event
sample scarcity?

## Evidence boundaries

- Report completed 24-hour and active 2-hour experiments as separate stories.
- Identify Surya as pretrained with parameter-efficient fine-tuning; do not
  generalize results to all foundation models.
- Use validation CSS for selection, then frozen selected checkpoints for test
  reporting.
- Make false-alarm statements from FAR/FPR, and uncertainty/model-comparison
  claims only after the planned temporal resampling analysis is run.

The planned tables and supplementary placement are specified in
[results reporting](../manuscript/results_reporting_plan.md).
