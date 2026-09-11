# Experiment Plan: Surya Flare Forecasting

> Historical planning record, superseded as the source of current experiment
> status by [research/experiment_plan.md](research/experiment_plan.md). The
> pending model matrix below captures earlier proposed work; it is not the
> current implementation or completion record.

## 1. Overview
This plan outlines the core experimental design for evaluating the pretrained solar foundation model **Surya** for solar flare forecasting, targeting Nature Scientific Reports.

## 2. Model Configurations (Priority Order)

| Configuration | Description | Status |
|---|---|---|
| **Model A: Conventional Baseline** | AlexNet (and optionally ResNet-18) trained from scratch to establish a supervised reference. | Pending |
| **Model B: Frozen Surya + Linear Head** | Extract Surya embeddings using frozen encoder, train linear classifier on cached embeddings. | Pending |
| **Model C: Frozen Surya + MLP Head** | Train a small MLP on cached Surya embeddings. | Pending |
| **Model D: Partial Fine-tuning** | Fine-tune prediction head + final block (and optionally final two blocks). | Pending |

## 3. Recommended Priority Order

1. **Priority 0: Verify Data Integrity**
   - Confirm split definitions, leakage, and pretraining overlap.
2. **Priority 1: Preserve Expensive Outputs**
   - Cache Surya embeddings (train/val/test).
   - Save validation and test probabilities.
3. **Priority 2: Required Surya Adaptations**
   - Implement Models B, C, and D.
4. **Priority 3: Baseline**
   - Train and preserve Model A.
5. **Priority 4: Statistical Evaluation**
   - Run multiple seeds, grouped bootstrap intervals, calibration, and probabilistic metrics.
6. **Priority 5: Learning Curves & Compute Trade-offs**
   - Test reduced training data (1%, 5%, 10%, 25%, 50%).
   - Measure resource usage vs performance.

## 4. Minimum Submission Package
- [ ] Conventional baseline
- [ ] Frozen Surya linear probe
- [ ] Frozen Surya MLP probe
- [ ] Partial fine-tuning
- [ ] Saved probabilities
- [ ] Validation-only threshold selection
- [ ] Confusion-matrix metrics, TSS, HSS, CSS, F1
- [ ] PR-AUC, ROC-AUC, Brier Score, Reliability Diagrams
- [ ] Multiple random seeds (min 5 for cached features, min 3 for fine-tuning)
- [ ] Grouped confidence intervals
- [ ] Computational cost reporting
