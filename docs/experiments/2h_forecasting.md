# 2-hour forecasting-window experiments

## Status

This is the current active experiment family. “Complete” below means a validation result is recorded locally, not that the experiment family has been finalized or that a Solar Cycle 25 test result exists. Do not select a final sampling interval or apply final best-configuration emphasis until the intended matrix is complete.

| Threshold | Sampling strategy | Sampling interval | Surya | AlexNet | ResNet18 |
| --- | --- | --- | --- | --- | --- |
| C>= | Full/non-undersampled | 24 h | Complete | Complete | Not evaluated |
| C>= | Full/non-undersampled | 12 h | Complete | Complete | Complete |
| C>= | Full/non-undersampled | 8 h | Complete | Complete | Complete |
| M>= | Undersampled | 2 h | Complete | Complete | Complete |
| M>= | Undersampled | 1 h | Complete | Complete | Complete |
| X>= | Undersampled | Not confirmed | Validation record, interval unrecorded | Validation record, interval unrecorded | Not evaluated |

The X>= row is deliberately not promoted to the main C>=/M>= matrix: the existing local table labels the run `daily_x2w` and says its sampling interval is unknown. A configuration file named `x2w_exp.yaml` does not resolve that provenance.

## Configuration evidence and limitations

Current training configurations reference C2w `freq8` for Surya and AlexNet,
and `freq24` for ResNet18; M2w `freq3` for Surya/ResNet18 and `freq2` for
AlexNet. These are current configuration settings, not a replacement for the
historical validation ledger labels above. Reconcile the run IDs, configuration
snapshots, and W&B history before final manuscript use.

### Configuration naming audit

- The M2w test configurations disable undersampling and use `nosample` labels,
  while their referenced checkpoint filenames say `undersample`; they also
  point at M2w checkpoint directories despite differing train-index intervals.
- The active ResNet18 M2w configuration applies undersampling but its checkpoint
  tag says `nosample`; its W&B name says `undersample`.
- C2w/X2w directory or W&B names use `daily` in places where the data index
  uses a frequency suffix. `daily` must not be used as evidence of the
  forecasting window or sampling interval.

Current explicit test configurations exist for Surya C/M/X, AlexNet C/M/X, and ResNet18 C/M; no ResNet18 X test configuration is present. Recent PBS test launchers invoke the M2w test configurations only. Neither fact demonstrates completed tests.

## Next reporting gate

For each intended cell, retain the validation checkpoint selected by CSS, then test only that checkpoint on Solar Cycle 25. Add final sampling-interval comparison and uncertainty only after the outstanding provenance and test records are audited.
