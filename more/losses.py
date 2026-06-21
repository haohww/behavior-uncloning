from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _target_mask(
    num_modes: int,
    target_modes: Iterable[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(num_modes, dtype=torch.bool, device=device)
    target = torch.as_tensor(list(target_modes), dtype=torch.long, device=device)
    if target.numel() == 0:
        raise ValueError("target_modes must contain at least one mode index")
    if torch.any(target < 0) or torch.any(target >= num_modes):
        raise ValueError("target mode index out of range")
    mask[target] = True
    return mask


def subset_redirect_loss(
    logits: torch.Tensor,
    target_modes: Iterable[int],
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Redirect classifier probability mass into a desired mode subset.

    This implements the paper's multi-target loss:

        L_red = -log sum_{k in S} softmax(logits)_k

    For a single target mode, it reduces to cross entropy. For a target subset,
    it increases the total probability assigned to the acceptable set.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, num_modes)")
    mask = _target_mask(logits.shape[-1], target_modes, logits.device)
    log_denom = torch.logsumexp(logits, dim=-1)
    neg_inf = torch.finfo(logits.dtype).min
    target_logits = logits.masked_fill(~mask[None, :], neg_inf)
    log_target_mass = torch.logsumexp(target_logits, dim=-1)
    per_sample = -(log_target_mass - log_denom)

    if weights is None:
        return per_sample.mean()
    if weights.shape != per_sample.shape:
        raise ValueError("weights must have shape (batch,)")
    denom = weights.sum().clamp_min(eps)
    return (per_sample * weights).sum() / denom


def source_mode_gate(
    logits: torch.Tensor,
    source_modes: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Return the MoRE source-mode gate.

    Undesired samples are redirected only while the frozen classifier assigns
    less than `tau` probability to the sample's source mode. This focuses
    editing on shared/pre-commit regions and avoids applying redirection after
    trajectories have strongly committed to the source behavior mode.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, num_modes)")
    if source_modes.shape != (logits.shape[0],):
        raise ValueError("source_modes must have shape (batch,)")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")

    probs = F.softmax(logits.detach(), dim=-1)
    src = source_modes.to(device=logits.device, dtype=torch.long)
    valid = (src >= 0) & (src < logits.shape[-1])
    safe_src = torch.where(valid, src, torch.zeros_like(src))
    p_source = probs.gather(1, safe_src[:, None]).squeeze(1)
    return ((p_source < tau) & valid).to(dtype=logits.dtype)


@dataclass
class MoreLossOutput:
    loss: torch.Tensor
    retain_loss: torch.Tensor
    redirect_loss: torch.Tensor
    gate_fraction: torch.Tensor


class MoreLoss(nn.Module):
    """Combine retain and classifier-guided redirect terms for MoRE editing."""

    def __init__(
        self,
        classifier: nn.Module,
        target_modes: Iterable[int],
        gamma: float = 1.0,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.target_modes = tuple(target_modes)
        self.gamma = gamma
        self.tau = tau
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        retain_loss: torch.Tensor,
        redirect_features: torch.Tensor,
        source_modes: torch.Tensor,
    ) -> MoreLossOutput:
        logits = self.classifier(redirect_features)
        gate = source_mode_gate(logits, source_modes, tau=self.tau)
        redirect_loss = subset_redirect_loss(logits, self.target_modes, weights=gate)
        total = retain_loss + self.gamma * redirect_loss
        return MoreLossOutput(
            loss=total,
            retain_loss=retain_loss.detach(),
            redirect_loss=redirect_loss.detach(),
            gate_fraction=(gate > 0).float().mean().detach(),
        )
