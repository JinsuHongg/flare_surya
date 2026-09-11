# Results reporting plan

## Validation Table 1 — 24-hour forecasting window

Panel (a) reports full/non-undersampled C>= and M>= experiments with HSS, TSS, and CSS. Panel (b) reports undersampled M>= and X>= experiments with the same metrics. For completed families, bold the highest CSS across sampling intervals for each model/task. The verified historical table currently supports Surya and AlexNet; it does not add a ResNet18 24-hour result retrospectively.

## Validation Table 2 — 2-hour forecasting window

Panel (a) reports C>= full/non-undersampled Surya, AlexNet, and ResNet18. Panel (b) reports M>= undersampled Surya, AlexNet, and ResNet18. Use HSS, TSS, and CSS. The family is ongoing: do not present a current maximum CSS as a final sampling-interval decision and do not apply final best-configuration bolding until all intended configurations are complete and provenance is verified.

Keep the uncertain X>= `daily_x2w` record out of the finalized main table until its sampling interval and run provenance are confirmed.

## Test reporting

Main Solar Cycle 25 test reporting should emphasize HSS, TSS, FAR, and POD, with a full-test estimate and planned temporal block-bootstrap interval. Put or retain F1-Macro, CSS, FPR, and raw confusion-matrix counts in supplementary material where they improve transparency. Do not make false-alarm claims from HSS alone.

