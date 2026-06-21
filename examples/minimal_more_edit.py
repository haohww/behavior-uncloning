from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from more import ModeClassifier, MoreTrainer, MoreTrainerConfig


class ToyPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def feature_fn(policy: ToyPolicy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Example differentiable classifier feature.

    Real integrations should replace this with the paper-specific feature:
    `(condition, predicted_clean_action_chunk)` for Diffusion Policy, or pooled
    VLA hidden state for pi0.5.
    """

    pred_action = policy(batch["obs"])
    return torch.cat([batch["obs"], pred_action], dim=-1)


def retain_loss_fn(policy: ToyPolicy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pred_action = policy(batch["obs"])
    return torch.nn.functional.mse_loss(pred_action, batch["action"])


def source_mode_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["mode"]


def main() -> None:
    torch.manual_seed(0)
    obs_dim = 6
    action_dim = 2
    num_modes = 2
    batch_size = 32

    policy = ToyPolicy(obs_dim, action_dim)
    classifier = ModeClassifier(input_dim=obs_dim + action_dim, num_modes=num_modes)

    # In a real run, train or load the classifier on original-policy features
    # before constructing MoreTrainer. This synthetic example only verifies the
    # editing plumbing.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    trainer = MoreTrainer(
        policy=policy,
        classifier=classifier,
        optimizer=optimizer,
        feature_fn=feature_fn,
        retain_loss_fn=retain_loss_fn,
        source_mode_fn=source_mode_fn,
        config=MoreTrainerConfig(target_modes=(0,), gamma=0.1, tau=0.5),
    )

    retain_batch = {
        "obs": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
        "mode": torch.zeros(batch_size, dtype=torch.long),
    }
    redirect_batch = {
        "obs": torch.randn(batch_size, obs_dim),
        "action": torch.randn(batch_size, action_dim),
        "mode": torch.ones(batch_size, dtype=torch.long),
    }

    stats = trainer.step(retain_batch, redirect_batch)
    print(stats)


if __name__ == "__main__":
    main()
