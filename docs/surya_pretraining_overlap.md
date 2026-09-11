# Surya Pretraining Overlap

*This document analyzes the overlap between Surya's pretraining data and the downstream flare forecasting test data.*

## 1. Pretraining Details
- **Surya Pretraining Date Range:** [TODO]
- **Instruments Used:** [TODO]
- **Wavelength Channels Used:** [TODO]

## 2. Overlap Assessment
- **Are downstream test images included in pretraining?** [TODO]
- **Are temporally adjacent observations included?** [TODO]
- **Are the same active regions included?** [TODO]

## 3. Mitigation Strategy
If exact overlap cannot be removed, the evaluation must accurately describe this. Do not claim a completely unseen temporal test if the model encountered the images during self-supervised pretraining.

Where feasible, create:
1. Complete future test set
2. Pretraining-nonoverlap subset
3. Active-region-disjoint subset
