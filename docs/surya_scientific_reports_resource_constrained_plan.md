# Surya Flare Forecasting: Resource-Constrained Publication Plan

> Historical planning record. Its resource and adaptation proposals remain
> useful context, but current implementation, experiment status, and evaluation
> decisions are maintained in `docs/research/` and `docs/experiments/`.

## 1. Project Goal

Evaluate whether the pretrained solar foundation model **Surya** provides useful and computationally efficient representations for solar flare forecasting.

The intended primary publication target is **Nature Scientific Reports**.

The study should not claim that Surya is universally superior or that full end-to-end fine-tuning establishes state-of-the-art performance. Instead, the paper should investigate:

> Does a pretrained solar foundation model improve flare forecasting when only limited labeled data and limited computational resources are available?

A second research question is:

> Which resource-efficient adaptation strategy provides the best trade-off between forecasting skill and computational cost?

---

## 2. Hard Constraints

### Data

- Input resolution: `4096 × 4096`
- Number of channels: `13`
- Task: solar flare forecasting
- Event thresholds may include:
  - `≥ C1.0`
  - `≥ M1.0`
  - `≥ X1.0`
- Forecast horizons currently include:
  - `2 hours`
  - `24 hours`
  - Additional horizons should only be added if computationally practical.

### Computational resources

- Full end-to-end Surya training is not computationally feasible.
- NASA computational resource access ends on **October 1, 2026**.
- Expensive Surya inference, feature extraction, and partial fine-tuning must be completed before that date.
- Lightweight downstream training and statistical analysis should be designed to run later on local resources.

### Publication constraints

The study should remain technically sound despite limited resources.

The paper must avoid unsupported claims such as:

- Surya always outperforms conventional models.
- Surya achieves state-of-the-art flare forecasting.
- Improvements are caused by pretraining when architecture effects have not been controlled.
- Results from a single random seed are statistically conclusive.

---

## 3. Recommended Paper Framing

### Preferred framing

**Resource-efficient adaptation of a solar foundation model for flare forecasting**

Alternative framing:

**Evaluation of pretrained Surya representations for flare forecasting under limited computational resources**

### Main contribution

The paper should evaluate whether frozen or partially adapted Surya representations can improve flare forecasting without computationally expensive full-model training.

### Possible contributions

1. Evaluation of Surya on C-, M-, and possibly X-class flare forecasting.
2. Comparison of frozen-feature, lightweight-head, and partial-fine-tuning strategies.
3. Analysis of forecasting performance under reduced labeled-data availability.
4. Comparison of forecasting skill against computational cost.
5. Evaluation under chronological distribution shift.
6. Analysis of conditions where Surya helps and where it does not.
7. Calibration and reliability analysis for operational flare forecasting.

---

## 4. Journal Recommendation

### Primary recommendation

Keep **Nature Scientific Reports** as the primary target for now.

The absence of full Surya fine-tuning does not automatically require changing journals. A resource-efficient adaptation study can be publishable if:

- the research question is clear,
- the comparison is fair,
- statistical uncertainty is reported,
- data leakage is carefully controlled,
- computational efficiency is measured,
- and the conclusions match the evidence.

### Consider another journal only if the final study remains too narrow

Reconsider the journal if the completed study contains only:

- one Surya configuration,
- one weak baseline,
- single-run metrics,
- no uncertainty estimates,
- no calibration analysis,
- no adaptation ablation,
- and no explanation of Surya failure cases.

Possible specialized fallback venues may include space-weather, solar-physics, geoscience, or remote-sensing journals. Journal selection should be revisited only after the minimum experiment package is complete.

---

## 5. Core Experimental Design

## 5.1 Required model configurations

Run the following configurations in priority order.

### Model A: Conventional supervised baseline

Use at least one conventional architecture:

- existing AlexNet baseline, and preferably
- ResNet-18 or another computationally manageable modern CNN.

Purpose:

- establish a standard supervised-learning reference,
- determine whether Surya features improve over a conventional model.

### Model B: Frozen Surya encoder with linear head

- Freeze all Surya encoder parameters.
- Extract Surya embeddings.
- Train a linear classifier on cached embeddings.

Purpose:

- measure the direct transferability of pretrained Surya representations,
- minimize training cost,
- allow repeated downstream experiments after NASA access ends.

### Model C: Frozen Surya encoder with MLP head

- Freeze all Surya encoder parameters.
- Train a small multilayer perceptron on cached embeddings.

Purpose:

- determine whether nonlinear adaptation improves over a linear probe,
- remain computationally inexpensive.

### Model D: Partial Surya fine-tuning

Fine-tune only a limited part of Surya.

Recommended variants:

1. prediction head only,
2. prediction head plus final Surya block,
3. prediction head plus final two Surya blocks, only if affordable.

Purpose:

- measure the benefit of limited representation adaptation,
- produce a performance-versus-compute curve.

### Optional Model E: Randomly initialized Surya

This is scientifically useful but may be computationally infeasible.

Do not prioritize it over feature extraction, partial fine-tuning, output preservation, and statistical validation.

If it cannot be run, state clearly that architecture and pretraining effects cannot be fully separated by training the complete Surya architecture from scratch.

---

## 5.2 Minimum comparison table

| Configuration | Encoder | Trainable components | Priority |
|---|---|---|---|
| AlexNet or ResNet | Randomly initialized | Full model | Required |
| Surya linear probe | Pretrained and frozen | Linear head | Required |
| Surya MLP probe | Pretrained and frozen | MLP head | Required |
| Surya partial fine-tuning | Pretrained | Final block and head | Required |
| Surya two-block fine-tuning | Pretrained | Final two blocks and head | Optional |
| Random Surya | Randomly initialized | Full model | Optional |

---

## 6. NASA Resource Usage Plan

## 6.1 Highest-priority task: cache Surya representations

Before **October 1, 2026**, extract and save Surya embeddings for every sample in:

- training set,
- validation set,
- test set.

Store sufficient metadata with every embedding:

- unique sample ID,
- observation timestamp,
- active-region ID if available,
- flare label,
- event threshold,
- forecast horizon,
- split assignment,
- input cadence or sampling configuration,
- Surya checkpoint version,
- encoder layer used for feature extraction,
- preprocessing configuration.

Recommended storage formats:

- HDF5,
- Zarr,
- Parquet plus array files,
- or PyTorch tensor files with a metadata table.

The format must support efficient partial loading.

### Why this is critical

Once embeddings are cached, the following can be performed without full Surya inference:

- linear probing,
- MLP training,
- repeated random seeds,
- class weighting,
- undersampling,
- threshold tuning,
- probability calibration,
- bootstrap confidence intervals,
- subgroup analysis,
- learning curves.

---

## 6.2 Save prediction outputs, not only final metrics

For every expensive run, save test and validation predictions.

Required fields:

- sample ID,
- true label,
- predicted probability,
- predicted binary label,
- selected decision threshold,
- model configuration,
- random seed,
- checkpoint,
- forecast horizon,
- flare threshold,
- active-region ID,
- timestamp.

These predictions are required for later calculation of:

- confusion matrices,
- ROC curves,
- precision–recall curves,
- Brier score,
- reliability diagrams,
- threshold sensitivity,
- confidence intervals,
- paired model comparisons.

Do not rely only on aggregate TSS, HSS, CSS, or F1 values.

---

## 6.3 Run expensive work only on NASA resources

Use NASA computing resources for:

- full-resolution Surya inference,
- Surya embedding extraction,
- partial Surya fine-tuning,
- final-block or two-block adaptation,
- generation of prediction probabilities from expensive models.

Do not spend limited NASA compute on tasks that can run locally, including:

- logistic regression,
- small MLP training on cached features,
- metric calculation,
- plotting,
- bootstrapping,
- threshold analysis,
- statistical testing,
- manuscript tables.

---

## 7. Evaluation Metrics

## 7.1 Deterministic forecasting metrics

Report at minimum:

- True positives
- False positives
- True negatives
- False negatives
- Precision
- Recall / probability of detection
- Specificity
- False-positive rate
- False-alarm ratio
- F1 score
- Macro F1 where appropriate
- True Skill Statistic
- Heidke Skill Score
- CSS, with an explicit definition

Do not report only skill scores.

---

## 7.2 Probabilistic forecasting metrics

Report:

- Brier score
- Brier skill score against a clearly defined climatology baseline
- Log loss
- ROC-AUC
- PR-AUC
- Reliability diagram
- Calibration slope and intercept, if practical
- Expected calibration error only as a supplementary metric

PR-AUC and reliability are particularly important for rare M- and X-class events.

---

## 7.3 Decision-threshold protocol

For every model:

1. Select the decision threshold using the validation set only.
2. Freeze the threshold.
3. Apply it once to the test set.
4. Report the selected threshold.
5. State the validation objective used to select it.

Possible validation objectives:

- maximum TSS,
- maximum CSS,
- maximum F1,
- or an operationally motivated false-alarm constraint.

Do not select thresholds using test-set performance.

Also produce:

- TSS versus threshold,
- HSS versus threshold,
- precision versus recall,
- confusion matrix at the frozen threshold.

---

## 8. Statistical Validation

## 8.1 Repeated runs

For cached-feature models:

- run at least `5` random seeds,
- preferably `10` if computationally inexpensive.

Report:

- mean,
- standard deviation,
- median where useful,
- 95% confidence interval.

Partial Surya fine-tuning may use fewer seeds if expensive, but aim for at least `3`.

Clearly distinguish:

- variability from downstream initialization,
- variability from data sampling,
- variability from threshold selection.

---

## 8.2 Grouped confidence intervals

Solar observations close in time are not independent.

Do not bootstrap individual frames as if they were independent.

Use grouped or block bootstrap based on the best available unit:

1. active region,
2. flare event,
3. observation day.

The preferred unit is the active region when active-region identifiers are available.

For each comparison, calculate a paired bootstrap interval for:

\[
\Delta M = M_{\text{Surya}} - M_{\text{baseline}},
\]

where \(M\) is TSS, HSS, CSS, F1, Brier score, or PR-AUC.

---

## 9. Leakage and Dataset Audit

The agent must audit and document the following.

### Dataset split

- Exact train years
- Exact validation years
- Exact test years
- Whether the split is chronological
- Whether the same active region appears in multiple splits
- Whether temporally adjacent observations cross split boundaries
- Whether observations linked to the same future flare appear in multiple splits
- Whether overlapping forecast windows create near-duplicate labels across splits

### Surya pretraining overlap

Document:

- Surya pretraining date range,
- instruments used,
- wavelength channels used,
- whether downstream test images were included in pretraining,
- whether temporally adjacent observations were included,
- whether the same active regions were included.

If exact overlap cannot be removed, describe the evaluation accurately. Do not claim a completely unseen temporal test when the model may have encountered the same images through self-supervised pretraining.

Where feasible, create:

1. complete future test set,
2. pretraining-nonoverlap subset,
3. active-region-disjoint subset.

---

## 10. Learning-Curve Experiment

A limited-data experiment is strongly recommended because it matches the foundation-model motivation.

Using cached Surya embeddings and the conventional baseline, train with:

- 1% of training data,
- 5%,
- 10%,
- 25%,
- 50%,
- 100%.

Sampling should preferably be grouped by active region rather than individual image.

Track:

- number of observations,
- number of unique active regions,
- number of positive flare events,
- class prevalence.

Plot:

- TSS versus number of training samples,
- PR-AUC versus number of positive events,
- Brier skill score versus training-set size.

The strongest evidence for Surya may be improved sample efficiency rather than the highest full-data score.

---

## 11. Computational-Efficiency Evaluation

For every model configuration, record:

- total parameter count,
- trainable parameter count,
- peak GPU memory,
- training GPU-hours,
- feature-extraction GPU-hours,
- inference time per sample,
- total inference time,
- storage size of cached embeddings,
- input resolution,
- number of channels,
- batch size,
- hardware model.

Create at least one resource-performance figure:

- TSS versus GPU-hours,
- PR-AUC versus trainable parameters,
- Brier skill score versus peak GPU memory,
- or performance versus total computational cost.

A possible supported conclusion is:

> Frozen or partially adapted Surya representations provide a favorable forecasting-skill-to-computation trade-off compared with conventional supervised training.

Make this claim only if the measured results support it.

---

## 12. Optional Scientific Analyses

Run these only after completing the required experiment package.

### Input-channel ablation

Possible groups:

- magnetogram only,
- EUV only,
- individual channels,
- selected channel subsets,
- all 13 channels.

### Forecast-horizon analysis

Potential horizons:

- 2 hours,
- 6 hours,
- 12 hours,
- 24 hours.

Do not add horizons unless label construction and computational cost are manageable.

### Solar-disk location

Evaluate separately for:

- central disk,
- intermediate longitude,
- near limb.

### Flare history

Evaluate:

- no recent flare,
- recent C-class flare,
- recent M/X-class flare.

Include a persistence baseline where possible.

### Yearly robustness

Report test performance by year to identify solar-cycle and prevalence effects.

---

## 13. Recommended Priority Order

### Priority 0: verify data integrity

- Confirm split definitions.
- Confirm no accidental train-test leakage.
- Document active-region overlap.
- Determine Surya pretraining overlap.

### Priority 1: preserve expensive outputs

- Cache Surya embeddings.
- Save validation probabilities.
- Save test probabilities.
- Save metadata and checkpoints.
- Verify that cached files can be loaded outside NASA infrastructure.

### Priority 2: required Surya adaptations

- Frozen Surya plus linear head.
- Frozen Surya plus MLP head.
- Final-block fine-tuning.
- Optional final-two-block fine-tuning.

### Priority 3: baseline

- Preserve the AlexNet baseline.
- Add ResNet-18 if feasible.
- Use the same split, labels, preprocessing, threshold protocol, and evaluation code.

### Priority 4: statistical evaluation

- Multiple seeds.
- Grouped bootstrap intervals.
- Paired model comparisons.
- Calibration and probabilistic metrics.

### Priority 5: learning curves and compute trade-offs

- Reduced training-data experiments.
- Trainable-parameter counts.
- GPU memory and GPU-hour reporting.
- Performance-versus-resource figures.

### Priority 6: optional subgroup analyses

- longitude,
- year,
- flare history,
- channel ablations,
- additional horizons.

---

## 14. Minimum Submission Package for Scientific Reports

The study should preferably include all of the following before submission:

- [ ] Clear resource-efficient adaptation research question
- [ ] Exact dataset and split documentation
- [ ] Active-region or temporal leakage audit
- [ ] Surya pretraining-overlap discussion
- [ ] Conventional baseline
- [ ] Frozen Surya linear probe
- [ ] Frozen Surya MLP probe
- [ ] At least one partial-fine-tuning configuration
- [ ] Saved validation and test probabilities
- [ ] Validation-only threshold selection
- [ ] Confusion-matrix metrics
- [ ] TSS, HSS, CSS, and F1
- [ ] PR-AUC and ROC-AUC
- [ ] Brier score or Brier skill score
- [ ] Reliability diagrams
- [ ] Multiple random seeds
- [ ] Grouped confidence intervals
- [ ] Computational-cost reporting
- [ ] Analysis of conditions where Surya underperforms
- [ ] Reproducible code and configuration files
- [ ] Data-availability statement
- [ ] Code-availability statement

---

## 15. Stop Criteria and Scope Control

Do not continue adding experiments indefinitely.

The project is ready to move into full manuscript writing when:

1. cached Surya embeddings and predictions are safely preserved,
2. the main baseline and three Surya adaptation strategies are evaluated,
3. uncertainty estimates are available,
4. calibration results are available,
5. leakage and pretraining overlap are documented,
6. computational cost is measured,
7. the main conclusion remains valid across the selected core tasks.

Do not spend scarce resources on:

- exhaustive hyperparameter searches,
- full Surya training from scratch,
- every possible flare threshold and horizon,
- weakly motivated architecture variants,
- repeated expensive runs whose outputs could have been obtained from cached embeddings.

---

## 16. Instructions for an AI Coding or Research Agent

The agent should:

1. Inspect the current repository, spreadsheet, configuration files, and saved outputs.
2. Create an inventory of completed and missing experiments.
3. Do not overwrite existing results.
4. Identify which tasks require NASA GPU resources and which can run locally.
5. Implement a robust Surya feature-caching pipeline.
6. Implement prediction-output saving with complete metadata.
7. Implement frozen linear and MLP probing.
8. Implement configurable partial fine-tuning.
9. Use shared evaluation code for all models.
10. Enforce validation-only threshold selection.
11. Implement grouped bootstrap confidence intervals.
12. Generate machine-readable result tables.
13. Generate publication-ready figures and LaTeX/Markdown tables.
14. Record hardware, runtime, memory, model size, and trainable parameters.
15. Maintain a run manifest containing configuration, seed, checkpoint, git commit, and output paths.
16. Flag potential data leakage rather than silently continuing.
17. Prefer completing the minimum submission package over adding optional experiments.

The agent should provide a proposed execution plan before launching expensive jobs and should estimate resource use from a small pilot run.

---

## 17. Expected Output Structure

Recommended repository structure:

```text
outputs/
├── embeddings/
│   ├── train/
│   ├── validation/
│   └── test/
├── predictions/
│   ├── alexnet/
│   ├── resnet18/
│   ├── surya_linear/
│   ├── surya_mlp/
│   └── surya_partial/
├── metrics/
│   ├── per_run.csv
│   ├── aggregate.csv
│   ├── confidence_intervals.csv
│   └── subgroup_metrics.csv
├── figures/
│   ├── performance/
│   ├── calibration/
│   ├── learning_curves/
│   └── compute_tradeoffs/
└── manifests/
    └── experiment_manifest.csv
```

Recommended documentation:

```text
docs/
├── experiment_plan.md
├── dataset_protocol.md
├── leakage_audit.md
├── surya_pretraining_overlap.md
├── evaluation_protocol.md
└── nasa_resource_plan.md
```

---

## 18. Final Recommendation

Do not change the journal target solely because full Surya training is infeasible.

Keep **Nature Scientific Reports** as the primary target and redesign the study around:

- frozen representation transfer,
- partial adaptation,
- sample efficiency,
- calibration,
- statistical robustness,
- and forecasting skill per unit of computation.

The immediate objective is not to maximize the number of experiments. It is to preserve expensive Surya outputs before **October 1, 2026** and complete a small, coherent, statistically defensible experiment set.
