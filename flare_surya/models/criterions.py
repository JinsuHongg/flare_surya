import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss from Lin et al. (ICCV 2017).
    Used in Alatoom & Nikolaou (2026) with alpha=0.5, gamma=2.0.

    Config example:
        loss:
          type: focal
          focal:
            alpha: 0.5   # Alatoom 2026 default (original Lin 2017 uses 0.25)
            gamma: 2.0
            reduction: mean
    """

    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        """
        alpha: weight for positive class (0.25 is common)
        gamma: focusing parameter (2.0 is common)
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: raw model output (before sigmoid), shape (N,)
        # targets: binary labels 0 or 1, shape (N,)

        # --- Label smoothing ---
        # Hard targets: 0 → 0.0, 1 → 1.0
        # Smoothed:     0 → ε/2,  1 → 1 - ε/2
        # For binary case we split ε equally across both classes (ε/2 each side)
        if self.label_smoothing > 0.0:
            targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FlareSSMLoss(nn.Module):
    """
    FLARE Loss from Takagi et al. (arXiv:2509.09988), adapted for binary classification.

    L = (L'_CE + L^IB_CE) + lambda_bss * (L'_BSS + L^IB_BSS)

    Requires hidden features h (the output of the penultimate layer, before the final
    FC classification layer). Use SuryaHead.forward_with_hidden() to obtain h.

    Args:
        class_counts: [N_negative, N_positive] — training sample counts per class,
                      used to compute inverse-frequency weights gamma(y).
        lambda_bss:   Weight for the BSS terms (paper default: 3.0).
        ib_start_epoch: IB losses are zero-ed out for epochs < this value,
                        for training stability (paper does not specify a value).
        eps:          Small constant to prevent division by zero.

    Forward signature: forward(logits, targets, h, current_epoch)
        logits:        (N, 1) — raw model output (before sigmoid)
        targets:       (N, 1) — binary labels {0, 1}
        h:             (N, D) — hidden features before the last FC layer
        current_epoch: int    — used to gate IB losses

    Config example:
        loss:
          type: flare
          flare:
            class_counts: [45000, 5000]   # [N_neg, N_pos] from training set
            lambda_bss: 3.0
            ib_start_epoch: 10
    """

    def __init__(
        self,
        class_counts: list,
        lambda_bss: float = 3.0,
        ib_start_epoch: int = 0,
        eps: float = 1e-8,
    ):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        # Inverse-frequency weights: gamma(y) = N_total / (K * N_k)
        total = counts.sum()
        n_classes = len(counts)
        gamma_weights = total / (n_classes * counts)
        self.register_buffer("gamma_weights", gamma_weights)
        self.lambda_bss = lambda_bss
        self.ib_start_epoch = ib_start_epoch
        self.eps = eps

    def _per_sample_gamma(self, targets: torch.Tensor) -> torch.Tensor:
        """Return gamma(y) for each sample. targets: (N, 1) with values 0/1."""
        idx = targets.long().squeeze(1)  # (N,)
        return self.gamma_weights.to(idx.device)[idx]  # (N,)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        h: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            logits:        (N, 1)
            targets:       (N, 1) — float, values in {0.0, 1.0}
            h:             (N, D) — hidden features before final FC
            current_epoch: current training epoch (for IB gating)
        Returns:
            Scalar loss.
        """
        p_hat = torch.sigmoid(logits)  # (N, 1)

        # Expand to 2-class representation to match the paper's multiclass notation.
        # For binary: p_hat_2 = [1-p, p],  y_onehot = [1-y, y]
        p_hat_2 = torch.cat([1.0 - p_hat, p_hat], dim=1)  # (N, 2)
        y_onehot = torch.cat([1.0 - targets, targets], dim=1)  # (N, 2)

        gamma = self._per_sample_gamma(targets)  # (N,)

        # ── Component 1: Weighted CE ──────────────────────────────────────────
        # Binary CE = -[y*log(p) + (1-y)*log(1-p)]
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).squeeze(
            1
        )  # (N,)
        l_ce_weighted = (gamma * bce).mean()

        # ── Shared norms ──────────────────────────────────────────────────────
        h_norm = h.abs().sum(dim=1).clamp(min=self.eps)  # (N,)
        # ||p_hat_2||_1 = sum of non-negative probs = 1.0 always (probability simplex)
        p_norm = p_hat_2.sum(dim=1).clamp(min=self.eps)  # (N,)  ≈ 1.0

        # ── Component 2: IB CE ───────────────────────────────────────────────
        ib_ce_denom = (p_norm * h_norm).clamp(min=self.eps)  # (N,)
        l_ib_ce = (gamma * bce / ib_ce_denom).mean()

        # ── BSS per sample: Σ_k (p_k - y_k)^2 ───────────────────────────────
        bss = ((p_hat_2 - y_onehot) ** 2).sum(dim=1)  # (N,)

        # ── Component 3: Weighted BSS ─────────────────────────────────────────
        l_bss_weighted = (gamma * bss).mean()

        # ── IB BSS denominator ────────────────────────────────────────────────
        # delta = p_hat_2 - y_onehot                           (N, 2)
        # dot_val = (delta ⊙ p_hat_2).sum(dim=1, keepdim=True) (N, 1)  scalar per sample
        # inner = delta - dot_val                              (N, 2)
        # ib_term = p_hat_2 ⊙ inner                           (N, 2)
        # ib_norm = ||ib_term||_1                              (N,)
        delta = p_hat_2 - y_onehot  # (N, 2)
        dot_val = (delta * p_hat_2).sum(dim=1, keepdim=True)  # (N, 1)
        inner = delta - dot_val  # (N, 2)
        ib_term = p_hat_2 * inner  # (N, 2)
        ib_norm = ib_term.abs().sum(dim=1).clamp(min=self.eps)  # (N,)
        ib_bss_denom = (2.0 * ib_norm * h_norm).clamp(min=self.eps)  # (N,)

        # ── Component 4: IB BSS ───────────────────────────────────────────────
        l_ib_bss = (gamma * bss / ib_bss_denom).mean()

        # ── IB gating (paper: IB losses off during early training) ────────────
        use_ib = current_epoch >= self.ib_start_epoch
        if not use_ib:
            l_ib_ce = torch.zeros_like(l_ce_weighted)
            l_ib_bss = torch.zeros_like(l_bss_weighted)

        return (l_ce_weighted + l_ib_ce) + self.lambda_bss * (l_bss_weighted + l_ib_bss)


def get_criterion(loss_dict, module_name=None):
    loss_type = loss_dict.get("type", "cross_entropy")
    match loss_type:
        case "cross_entropy":
            ce_dict = loss_dict.get("cross_entropy", {})
            class_weights = ce_dict.get("class_weights", None)
            if class_weights is not None:
                # Binary case: pos_weight = weight_pos / weight_neg
                pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return nn.BCEWithLogitsLoss()
        case "focal":
            return BinaryFocalLoss(
                alpha=loss_dict["focal"].get("alpha", 0.25),
                gamma=loss_dict["focal"].get("gamma", 2.0),
                reduction=loss_dict["focal"].get("reduction", "mean"),
                label_smoothing=loss_dict["focal"].get("label_smoothing", 0.0),
            )
        case "flaressm":
            if module_name == "BaseLineModel":
                raise ValueError(
                    "FlareSSMLoss requires hidden features (h) from the penultimate layer "
                    "and is only supported for FlareSurya, not BaseLineModel."
                )
            flaressm_cfg = loss_dict.get("flaressm", {})
            return FlareSSMLoss(
                class_counts=list(flaressm_cfg.get("class_counts", [1, 1])),
                lambda_bss=flaressm_cfg.get("lambda_bss", 3.0),
                ib_start_epoch=flaressm_cfg.get("ib_start_epoch", 0),
            )
        case _:
            raise ValueError(f"Unsupported loss type: {loss_type}")
