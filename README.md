# flare-surya

`flare-surya` evaluates the Surya heliophysics foundation model for imbalanced
solar-flare forecasting, alongside AlexNet and ResNet18 supervised baselines.
The current study is being prepared for *Scientific Reports* and examines when
a foundation-model advantage persists—and when rare-event sample scarcity
limits it.

The experiments use the [Surya Bench flare-forecasting dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting).
Surya input imagery is provided separately through the associated Surya data
release. This repository does not include the dataset, checkpoints, W&B runs,
or generated results.

## Experiment organization

- `configs/nas/surya/`: Surya fine-tuning configurations.
- `configs/nas/baselines/`: AlexNet and ResNet18 configurations.
- `configs/nas/exp_surya.yaml` and `configs/nas/baselines_exp.yaml`: shared
  Hydra defaults.
- `scripts/finetuning/`: Surya training and test entry points.
- `scripts/training/`: baseline training and test entry points.
- `shell_scripts/`: PBS launch scripts.

The study separates a 24-hour forecasting-window historical experiment family
from an active 2-hour forecasting-window family. “Forecasting window” is the
prediction horizon; “sampling interval” is the frequency used to extract
timeline samples. See [the current research state](docs/research/current_state.md),
[24-hour experiments](docs/experiments/24h_forecasting.md), and
[2-hour experiments](docs/experiments/2h_forecasting.md) for the evidence and
status matrix.

## Running configured work

With the project environment and data paths available, launch Hydra-configured
Surya jobs with:

```bash
python scripts/finetuning/finetuning.py +surya=c2w_exp
```

Launch a baseline job with:

```bash
python scripts/training/training_baseline.py +baselines=resnet18_c2w
```

Use a test configuration (for example, `+surya=test_m_run` or
`+baselines=resnet18_m_test`) only after verifying its referenced checkpoint
and test index. PBS launch scripts under `shell_scripts/nas/` encode the
cluster execution environment.

Large and local artifacts—including `data/`, `results/`, checkpoints, W&B
files, logs, Zarr stores, and CSV outputs—are intentionally ignored. Project
documentation under `docs/` is version controlled.
