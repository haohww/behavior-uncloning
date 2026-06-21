from __future__ import annotations

import torch
from torch import nn


class ModeClassifier(nn.Module):
    """Small K-way mode classifier used by MoRE.

    The classifier is trained on features from the original mixed policy and is
    frozen during policy editing. Its input is policy-specific:

    - Diffusion Policy: concatenate the policy condition and predicted clean
      action chunk.
    - VLA: use the pooled hidden state described in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int,
        hidden_dim: int = 256,
        num_hidden_blocks: int = 3,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_modes <= 1:
            raise ValueError("num_modes must be at least 2")

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_hidden_blocks):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_modes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    @torch.no_grad()
    def predict_modes(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).argmax(dim=-1)
