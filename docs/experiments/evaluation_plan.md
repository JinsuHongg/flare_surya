# Evaluation plan

## Implemented

The classification metric module accumulates TP, FP, TN, and FN at the fixed configured threshold (currently 0.5) and computes HSS, TSS, CSS, accuracy, precision, recall, and F1. Test execution saves timestamp/probability/target rows when `save_test_results_path` is configured, logs aggregate counts to W&B, and produces ROC/PR and threshold-sweep W&B artifacts. The retrieval script derives FAR and POD from aggregate W&B counts.

The validation retrieval script selects the highest recorded `val/css` epoch:

`CSS = sqrt(TSS * HSS)`

with the implementation clamping negative HSS/TSS to zero before the square root. For paper selection, use the highest validation CSS within a model plus sampling-interval configuration, then compare retained checkpoints across sampling intervals.

The Lightning checkpoint callbacks instead monitor the configuration's `optimizer.scheduler.monitor`. The Surya NAS default is `val/css`, whereas the baseline NAS default is `val/hss`; therefore callback retention is not alone evidence of CSS selection for every run. Audit W&B history and checkpoint identity before the final table.

## Test metrics

Primary Solar Cycle 25 metrics are HSS, TSS, FAR, and POD:

- `FAR = FP / (TP + FP)`
- `POD = TP / (TP + FN)`
- `FPR = FP / (FP + TN)`

FPR is an additional/supplementary metric. Report raw TP, FP, TN, and FN where helpful for auditability. F1-Macro and CSS belong in supplementary reporting unless a specific analysis needs them.

## Planned, not implemented

Solar Cycle 25 uncertainty should report a full-test point estimate and a temporal block-bootstrap 95% confidence interval. Do not use a naive sample-wise IID bootstrap as the primary method: hourly examples can have overlapping prediction windows and are temporally dependent.

Initial plan: approximately 7-day temporal blocks, approximately 1,000–5,000 replicates, and a block-length sensitivity analysis if feasible. Compare models using a **paired** temporal block bootstrap: in every replicate, all compared models use the same sampled temporal blocks and report paired differences (`ΔHSS`, `ΔTSS`, `ΔFAR`, and `ΔPOD`). No bootstrap implementation was found.

Saved per-example timestamp/probability/target files could support this work if retained. They do not currently include explicit run metadata, binary predictions, threshold, sample IDs, or active-region/event identifiers; aggregate W&B confusion counts alone cannot support the bootstrap.

