# Results reporting plan

Use the [evaluation plan](../experiments/evaluation_plan.md#standard-score-reporting-protocol)
as the canonical source for score definitions, directions, checkpoint selection,
and the validation-selected probability threshold applied to test results.
Validation and test audit tables use the same nine score columns; leave missing
test results blank. The manuscript's primary test table may present the
prespecified subset below.

## Validation Table 1 — 24-hour forecasting window

Panel (a) reports full/non-undersampled C>= and M>= experiments. Panel (b) reports undersampled M>= and X>= experiments. Include the complete validation score set and selected probability threshold in the audit table; the concise manuscript table may emphasize HSS, TSS, and CSS. For completed families, bold the highest CSS across sampling intervals for each model/task. The verified historical table currently supports Surya and AlexNet; it does not add a ResNet18 24-hour result retrospectively.

## Validation Table 2 — 2-hour forecasting window

Panel (a) reports C>= full/non-undersampled Surya, AlexNet, and ResNet18. Panel (b) reports M>= undersampled Surya, AlexNet, and ResNet18. Include the complete validation score set and selected probability threshold in the audit table; the concise manuscript table may emphasize HSS, TSS, and CSS. The family is ongoing: do not present a current maximum CSS as a final sampling-interval decision and do not apply final best-configuration bolding until all intended configurations are complete and provenance is verified.

Keep the uncertain X>= `daily_x2w` record out of the finalized main table until its sampling interval and run provenance are confirmed.

## Test reporting

Main Solar Cycle 25 test reporting should emphasize HSS, TSS, FAR, and POD, with a full-test estimate and a temporal block-bootstrap interval only when computed. Apply the matching validation-selected probability threshold; never choose the operating threshold from test performance. Put the complete nine-score set, selected threshold, FPR, and useful raw confusion-matrix counts in the audit/supplementary material. Do not make false-alarm claims from HSS alone.

