# 24-hour forecasting-window experiments

## Status and scope

This is the completed historical **validation** experiment family. The local manuscript validation records support the statuses below for Surya and AlexNet; they do not establish that every corresponding Solar Cycle 25 test was run. ResNet18 is implemented but has no verified 24-hour result record in the local table.

## Full/non-undersampled training

| Prediction target | Sampling interval | Surya | AlexNet | ResNet18 |
| --- | --- | --- | --- | --- |
| C>= | 24 h, 12 h, 8 h | Complete (validation record) | Complete (validation record) | Not evaluated in historical record |
| M>= | 24 h, 12 h, 8 h | Complete (validation record) | Complete (validation record) | Not evaluated in historical record |

## Undersampled training

| Prediction target | Sampling interval | Surya | AlexNet | ResNet18 |
| --- | --- | --- | --- | --- |
| M>= | 4 h, 3 h, 2 h | Complete (validation record) | Complete (validation record) | Not evaluated in historical record |
| M>= | 1 h | Not evaluated in historical record | Not evaluated in historical record | Not evaluated |
| X>= | 4 h, 3 h, 2 h, 1 h | Complete (validation record) | Complete (validation record) | Not evaluated in historical record |

The current NAS YAMLs each describe one active configuration (for example, `c_exp.yaml` presently points to C24w `freq24`, while `m_exp.yaml` points to M24w `freq12`). They should not be read as a complete enumeration of the historical sweep.

## Selection and test separation

Training configurations define the data index, undersampling flag/factor, model, optimizer, and checkpoint directory. Validation selection should retain the checkpoint with highest CSS within a model plus sampling interval. Solar Cycle 25 evaluation is a separate test invocation with an explicit checkpoint and test index. Existing test YAMLs are candidates for that procedure, not proof of completed test evaluation.

See [evaluation plan](evaluation_plan.md) for metrics and statistical analysis.

