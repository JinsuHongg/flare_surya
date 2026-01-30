import torch
from torchmetrics import Metric


class DistributedClassificationMetrics(Metric):
    """
    Computes global classification metrics (Accuracy, Precision, Recall, F1, TSS, HSS)
    in a distributed (DDP) setting.

    Args:
        threshold: Probability threshold for converting logits/probs to binary (default 0.5).
        dist_sync_on_step: Synchronize metric state across processes at each `forward()`.
                           False improves speed (default).
    """

    # 1. Define state variables (these are automatically synced across GPUs)
    tp: torch.Tensor
    tn: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor

    def __init__(self, threshold: float = 0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold

        # 2. Register states using add_state()
        # dist_reduce_fx="sum" tells Lightning to SUM these values across all GPUs
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        preds: Probabilities (0.0 to 1.0) or Logits
        target: Ground truth (0 or 1)
        """
        # Ensure predictions are binary (0 or 1)
        # If you pass logits, apply sigmoid before passing to update, or handle it here
        preds = (preds > self.threshold).long()
        target = target.long()

        # Update states (Vectorized for speed)
        self.tp += (preds * target).sum()
        self.tn += ((1 - preds) * (1 - target)).sum()
        self.fp += (preds * (1 - target)).sum()
        self.fn += ((1 - preds) * target).sum()

    def compute(self):
        """
        Compute final metrics from the accumulated states.
        Automatic float casting prevents integer division issues.
        """
        # Use epsilon to prevent DivisionByZero (NaNs)
        eps = 1e-7

        # 1. Base Metrics
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)

        # 2. F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        # 3. TSS (True Skill Statistic) = Sensitivity + Specificity - 1
        # Specificity = TN / (TN + FP)
        specificity = self.tn / (self.tn + self.fp + eps)
        tss = recall + specificity - 1

        # 4. HSS (Heidke Skill Score)
        numerator = 2 * (self.tp * self.tn - self.fn * self.fp)
        denominator = (self.tp + self.fn) * (self.fn + self.tn) + (
            self.tp + self.fp
        ) * (self.tn + self.fp)
        hss = numerator / (denominator + eps)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tss": tss,
            "hss": hss,
            # Raw counts can be useful for debugging
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }
