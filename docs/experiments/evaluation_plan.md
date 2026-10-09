# Evaluation plan

## Implemented

The classification metric module accumulates TP, FP, TN, and FN at the fixed configured threshold (currently 0.5) and computes HSS, TSS, CSS, accuracy, precision, recall, and F1. Test execution saves timestamp/probability/target rows when `save_test_results_path` is configured, logs aggregate counts to W&B, and produces ROC/PR and threshold-sweep W&B artifacts. The retrieval script derives FAR and POD from aggregate W&B counts.

The validation retrieval script selects the highest recorded `val/css` epoch:

`CSS = sqrt(TSS * HSS)`

with the implementation clamping negative HSS/TSS to zero before the square root. For paper selection, use the highest validation CSS within a model plus sampling-interval configuration, then compare retained checkpoints across sampling intervals.

The Lightning checkpoint callbacks instead monitor the configuration's `optimizer.scheduler.monitor`. The Surya NAS default is `val/css`, whereas the baseline NAS default is `val/hss`; therefore callback retention is not alone evidence of CSS selection for every run. Audit W&B history and checkpoint identity before the final table.

## Standard score-reporting protocol

Use this protocol for validation and test audit tables so results remain
comparable across forecasting windows and future experiment updates.

### Checkpoint and probability-threshold selection

1. Within each experiment configuration, select the checkpoint from the
   highest recorded validation CSS epoch. If W&B contains multiple records for
   an epoch, use the latest record for that epoch. Verify the checkpoint/run
   identity; do not infer that the callback-retained checkpoint is the CSS
   maximum.
2. For each model, flare-label threshold, forecasting window, sampling
   interval, and training/sampling setup, select the probability threshold that
   maximizes CSS on the selected checkpoint's per-sample validation
   predictions. Use the existing 0.01–0.99 grid in 0.01 increments, keep that
   grid identical for compared models, and store it with the selected
   threshold. If multiple grid values tie for maximum CSS, record the tie and
   apply a tie rule documented before test evaluation. Do not use test labels
   to select a threshold.
3. Apply that frozen validation threshold to the matching test experiment
   (same model, flare-label threshold, forecasting window, sampling interval,
   and training/sampling setup). If the matching validation threshold or test
   result is unavailable, leave the corresponding test cells blank; never
   substitute zero or optimize on the test set.

The metric module's default threshold is 0.5, but that fixed-threshold output
is not the CSS-optimized result. Threshold-sweep curves computed on test data
are descriptive only and must not determine the reported test operating point.

Before calculating scores, verify one prediction row per sample. Exact
duplicate rows may be reduced to one. If a sample is repeated with different
probabilities or labels, resolve the inference-pass provenance or rerun
inference; do not average the rows or count both as independent samples. For
the current prediction CSVs, use the full `timestamps` tuple as the sample key.

### Scores and definitions

The standard validation and test audit tables report the same score columns:
HSS, TSS, CSS, FAR, POD, Macro F1, Positive F1, PR-AUC, and Brier Score.
Show the selected probability threshold alongside the results. The first seven
classification scores use that threshold; PR-AUC and Brier Score use the
probabilities directly and do not depend on a decision threshold.

Let TP, FP, TN, and FN be counts at the selected probability threshold. A
sample is predicted positive when its probability is strictly greater than
the threshold, matching the current metric implementation.

| Score | Definition | Preferred direction |
| --- | --- | --- |
| HSS | `2(TP·TN − FN·FP) / [(TP+FN)(FN+TN) + (TP+FP)(TN+FP)]` | Higher |
| TSS | `TP/(TP+FN) − FP/(FP+TN)` | Higher |
| CSS | `sqrt(max(HSS, 0) · max(TSS, 0))` | Higher |
| FAR | `FP/(TP+FP)` | Lower |
| POD | `TP/(TP+FN)` | Higher |
| Positive F1 | `2TP/(2TP+FP+FN)` | Higher |
| Macro F1 | Mean of positive-class F1 and negative-class F1 | Higher |
| PR-AUC | Average precision (AP) from the per-sample probabilities and labels | Higher |
| Brier Score | Mean squared error between probability and binary label | Lower |

Use the same probability/label rows for PR-AUC and Brier Score as for the
thresholded scores. Do not round probabilities before calculating scores.
Keep unavailable metrics blank and report the selected threshold and its
selection split. In tables, mark higher-is-better scores with `↑` and
lower-is-better scores with `↓`.

For the main manuscript test table, emphasize HSS, TSS, FAR, and POD. Retain
the complete nine-score set, selected threshold, and useful confusion-matrix
counts in the audit/supplementary table. Report a 95% confidence interval only
when it has actually been computed; identify the resampling unit and method.

## Test metrics

Primary Solar Cycle 25 metrics are HSS, TSS, FAR, and POD:

- `FAR = FP / (TP + FP)`
- `POD = TP / (TP + FN)`
- `FPR = FP / (FP + TN)`

FPR is an additional/supplementary metric. Report raw TP, FP, TN, and FN where helpful for auditability. The full score set and operating-threshold rule are specified above.

## Planned, not implemented

Solar Cycle 25 uncertainty should report a full-test point estimate and a temporal block-bootstrap 95% confidence interval. Do not use a naive sample-wise IID bootstrap as the primary method: hourly examples can have overlapping prediction windows and are temporally dependent.

Initial plan: approximately 7-day temporal blocks, approximately 1,000–5,000 replicates, and a block-length sensitivity analysis if feasible. Compare models using a **paired** temporal block bootstrap: in every replicate, all compared models use the same sampled temporal blocks and report paired differences (`ΔHSS`, `ΔTSS`, `ΔFAR`, and `ΔPOD`). No bootstrap implementation was found.

Saved per-example timestamp/probability/target files could support this work if retained. They do not currently include explicit run metadata, binary predictions, threshold, sample IDs, or active-region/event identifiers; aggregate W&B confusion counts alone cannot support the bootstrap.

