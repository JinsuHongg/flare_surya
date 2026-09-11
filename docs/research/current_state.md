# Current research state

## Research goal

This project evaluates the Surya heliophysics foundation model for highly
imbalanced solar flare forecasting for a *Scientific Reports* manuscript. The
working framing is: **systematically evaluate when a heliophysics foundation
model improves imbalanced solar flare forecasting and when its advantage is
limited by rare-event sample scarcity.** It does not assume that foundation
models solve class imbalance.

## Implemented models

- **Surya:** `HelioSpectFormer` loaded with Surya weights, an MLP prediction
  head, and configurable LoRA parameter-efficient fine-tuning. The shared NAS
  configuration freezes the backbone and enables LoRA; individual experiments
  inherit those defaults unless overridden.
- **AlexNet:** a supervised classifier supplied by `terratorch-surya` and
  trained through `BaseLineModel`.
- **ResNet18:** a supervised `terratorch-surya` baseline. At construction, its
  `BatchNorm2d` layers are recursively replaced with GroupNorm (default eight
  groups), including in the current baseline entry point.

The code imports ResNet34 and ResNet50 alternatives, but no current NAS
experiment configuration targets them; they are not part of the paper matrix.

## Experiment families

| Forecasting window | Thresholds and training strategy | Sampling intervals | State |
| --- | --- | --- | --- |
| 24 h | C>=, M>= full/non-undersampled | 24 h, 12 h, 8 h | Historical completed validation records for Surya and AlexNet |
| 24 h | M>= undersampled | 4 h, 3 h, 2 h | Historical completed validation records for Surya and AlexNet |
| 24 h | X>= undersampled | 4 h, 3 h, 2 h, 1 h | Historical completed validation records for Surya and AlexNet |
| 2 h | C>= full/non-undersampled | 24 h, 12 h, 8 h | Active; documented records exist, but the family is not final |
| 2 h | M>= undersampled | 2 h, 1 h | Active; documented records exist, but the family is not final |
| 2 h | X>= undersampled | configuration exists; sampling interval/result status needs confirmation | Not a finalized paper family |

“Forecasting window” means prediction horizon (24 h or 2 h). “Sampling
interval” means the interval used to extract samples from the timeline. These
terms must not be conflated with cadence, temporal resolution, or input window.

The status statements above combine configuration inspection with the existing
local validation-result records. A YAML file alone is never evidence that an
experiment completed. Details are in the experiment documents.

## Current implementation and analysis capability

At threshold 0.5, the distributed metric code computes accuracy, precision,
recall, F1, TSS, HSS, CSS, and TP/TN/FP/FN. CSS is
`sqrt(max(HSS, 0) * max(TSS, 0))`. Test steps retain timestamps, probabilities,
and targets and flush them to the configured CSV path; W&B also receives ROC,
PR, and threshold-sweep artifacts. W&B retrieval scripts select the validation
epoch with maximum `val/css` and retrieve aggregate test counts.

Per-example test CSVs can support FAR, POD, FPR, and temporal bootstrap work
when they have actually been saved and retained. Aggregate W&B counts support
point estimates only; they are not sufficient for a temporal bootstrap.

## Scope cautions

The root-level legacy plans record earlier ideas (frozen probes, cached
representations, broad calibration, repeated seeds, and compute accounting).
They remain useful history, but they are not evidence that those analyses are
implemented or complete. See [decision log](decision_log.md).
