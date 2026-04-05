# New Loss Functions

Two additional loss functions for solar flare prediction: **Focal Loss** and **FLARE Loss**.

## Config Usage

```yaml
# Default (existing loss)
loss:
  type: ce

# Focal Loss
loss:
  type: focal
  focal:
    alpha: 0.5
    gamma: 2.0

# FLARE Loss
loss:
  type: flare
  flare:
    class_counts: [45000, 5000]  # [N_neg, N_pos] from your training set
    lambda_bss: 3.0
    ib_start_epoch: 10
```

`class_counts` must be set manually based on your training data.

## Focal Loss

From Lin et al., ICCV 2017. Down-weights easy samples, focuses on hard examples.

| Parameter | Default | Range to try |
|-----------|---------|--------------|
| `focal.gamma` | 2.0 | 0.5 – 5.0 |
| `focal.alpha` | 0.5 | 0.25 – 0.75 |

## FLARE Loss

From Takagi et al., arXiv:2509.09988. Composite loss with influence-balanced weighting for class imbalance.

| Parameter | Default | Range to try | Notes |
|-----------|---------|--------------|-------|
| `flare.lambda_bss` | 3.0 | 1.0 – 5.0 | Higher = more emphasis on probability calibration |
| `flare.ib_start_epoch` | 10 | ~20-30% of total epochs | Too early → unstable; too late → no effect |
| `flare.class_counts` | — | — | Required. Count of [negative, positive] samples in training set |

**Note:** FLARE Loss requires hidden features from the penultimate layer. Only works with `FlareSurya` model (not `BaseLineModel`).

## Files Changed

- `criterions.py` — `FLARELoss` class added
- `heads.py` — `SuryaHead.forward_with_hidden()` added (returns logits + hidden features)
- `modules.py` — loss selection and training step updated

