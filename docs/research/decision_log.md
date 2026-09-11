# Decision log

## Current decisions

- **2026-09-07 — Journal target:** *Scientific Reports* is the current target
  journal.
- **2026-09-07 — Checkpoint selection:** choose the highest validation CSS
  within each model and sampling-interval configuration.
- **2026-09-07 — Validation reporting:** retain CSS in validation tables
  because it is the checkpoint-selection criterion.
- **2026-09-07 — Main test metrics:** emphasize HSS, TSS, FAR, and POD.
  F1-Macro, CSS, FPR, and raw confusion-matrix counts are supplementary where
  appropriate.
- **2026-09-07 — False alarms:** claims about false alarms require direct FAR
  or FPR evidence, not HSS alone.
- **2026-09-07 — Test uncertainty:** Solar Cycle 25 confidence intervals
  should use temporal/block-aware resampling, not naive IID resampling.
- **2026-09-07 — Model comparisons:** use paired temporal block bootstrap:
  every compared model receives the same resampled temporal blocks.
- **2026-08-15 — Stronger baseline:** ResNet18 was added and its BatchNorm2d
  layers are converted to GroupNorm in the current baseline model path.
- **2026-09-07 — Terminology:** sampling interval and forecasting window are
  separate concepts.
- **2026-09-07 — Narrative:** 24-hour and 2-hour forecasting-window results
  are separate experiment stories.

## Historical or superseded planning material

The root-level `docs/experiment_plan.md`, `evaluation_protocol.md`, and
`surya_scientific_reports_resource_constrained_plan.md` preserve useful earlier
ideas. Their frozen-probe, broad probabilistic-metric, repeated-seed, cached
embedding, and resource-accounting items are historical proposals unless the
code and retained experiment record demonstrate completion. They do not
override this log or the current experiment/evaluation documents.
