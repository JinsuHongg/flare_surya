# Evaluation Protocol

> Historical planning record. The code-backed current policy is in
> [experiments/evaluation_plan.md](experiments/evaluation_plan.md). Items below
> are proposals unless the implementation and retained experiment record show
> otherwise.

## 1. Deterministic Metrics
Report the following at minimum:
- True positives, False positives, True negatives, False negatives
- Precision, Recall (Probability of Detection), Specificity
- False-positive rate, False-alarm ratio
- F1 score (and Macro F1 where appropriate)
- True Skill Statistic (TSS)
- Heidke Skill Score (HSS)
- CSS (with explicit definition)

## 2. Probabilistic Metrics
Particularly important for rare M- and X-class events:
- Brier Score
- Brier Skill Score (against a clearly defined climatology baseline)
- Log loss
- ROC-AUC
- PR-AUC
- Reliability diagram
- Calibration slope and intercept (if practical)
- *(Expected calibration error only as a supplementary metric)*

## 3. Decision-Threshold Protocol
1. Select the decision threshold using the **validation set only** (e.g., maximizing TSS, CSS, F1, or constrained by false-alarm rate).
2. Freeze the threshold.
3. Apply it **once** to the test set.
4. Report the selected threshold and the validation objective used to select it.
5. Produce curves: TSS vs threshold, HSS vs threshold, Precision vs Recall, and Confusion Matrix at frozen threshold.

## 4. Statistical Validation
- **Repeated Runs:** Minimum 5 random seeds (preferably 10) for cached-feature models; minimum 3 for partial fine-tuning.
- **Grouped Confidence Intervals:** Use grouped or block bootstrap based on the best available unit (Active region > Flare event > Observation day). Calculate paired bootstrap intervals for metrics like TSS, HSS, CSS, F1, Brier score, or PR-AUC.

## 5. Computational-Efficiency Evaluation
Record for every model configuration:
- Total / Trainable parameter count
- Peak GPU memory
- Training / Feature-extraction GPU-hours
- Inference time
- Storage size of cached embeddings
