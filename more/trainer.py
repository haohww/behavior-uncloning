from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn

from .losses import MoreLoss


Batch = object
FeatureFn = Callable[[nn.Module, Batch], torch.Tensor]
RetainLossFn = Callable[[nn.Module, Batch], torch.Tensor]
SourceModeFn = Callable[[Batch], torch.Tensor]


@dataclass
class MoreTrainerConfig:
    target_modes: tuple[int, ...]
    gamma: float = 1.0
    tau: float = 0.5
    grad_clip_norm: float | None = 1.0


class MoreTrainer:
    """Policy-agnostic MoRE training step.

    The trainer deliberately keeps policy-specific details outside the package.
    A Diffusion Policy integration should make `feature_fn` return the
    differentiable `(condition, predicted_clean_action_chunk)` features. A VLA
    integration should return the differentiable pooled hidden state.
    """

    def __init__(
        self,
        policy: nn.Module,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        feature_fn: FeatureFn,
        retain_loss_fn: RetainLossFn,
        source_mode_fn: SourceModeFn,
        config: MoreTrainerConfig,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.feature_fn = feature_fn
        self.retain_loss_fn = retain_loss_fn
        self.source_mode_fn = source_mode_fn
        self.loss_fn = MoreLoss(
            classifier=classifier,
            target_modes=config.target_modes,
            gamma=config.gamma,
            tau=config.tau,
        )
        self.grad_clip_norm = config.grad_clip_norm

    def step(self, retain_batch: Batch, redirect_batch: Batch) -> dict[str, float]:
        self.policy.train()
        retain_loss = self.retain_loss_fn(self.policy, retain_batch)
        redirect_features = self.feature_fn(self.policy, redirect_batch)
        source_modes = self.source_mode_fn(redirect_batch)

        out = self.loss_fn(
            retain_loss=retain_loss,
            redirect_features=redirect_features,
            source_modes=source_modes,
        )

        self.optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(out.loss.detach()),
            "retain_loss": float(out.retain_loss),
            "redirect_loss": float(out.redirect_loss),
            "gate_fraction": float(out.gate_fraction),
        }


def as_target_tuple(target_modes: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(mode) for mode in target_modes)
