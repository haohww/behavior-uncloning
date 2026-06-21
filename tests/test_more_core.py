from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from more import (
    ModeClassifier,
    MoreLoss,
    diffusion_policy_features,
    source_mode_gate,
    subset_redirect_loss,
)


class MoreCoreTest(unittest.TestCase):
    def test_subset_redirect_loss_matches_manual_probability(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
        loss = subset_redirect_loss(logits, target_modes=(0, 2))

        probs = torch.softmax(logits, dim=-1)
        expected = -torch.log(probs[:, [0, 2]].sum(dim=-1)).mean()
        self.assertTrue(torch.allclose(loss, expected))

    def test_subset_redirect_loss_supports_sample_weights(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        weights = torch.tensor([1.0, 0.0, 1.0])
        loss = subset_redirect_loss(logits, target_modes=(0,), weights=weights)

        per_sample = -torch.log(torch.softmax(logits, dim=-1)[:, 0])
        expected = (per_sample * weights).sum() / weights.sum()
        self.assertTrue(torch.allclose(loss, expected))

    def test_subset_redirect_loss_is_not_probability_capped(self) -> None:
        logits = torch.tensor([[1000.0, -1000.0]])
        loss = subset_redirect_loss(logits, target_modes=(1,))
        self.assertTrue(torch.allclose(loss, torch.tensor(2000.0)))

    def test_source_mode_gate_uses_source_probability(self) -> None:
        logits = torch.tensor(
            [
                [0.0, 3.0],  # source 1 is confident: do not redirect
                [3.0, 0.0],  # source 1 is not confident: redirect
                [1.0, 1.0],  # exactly 0.5 source probability: strict < tau
            ]
        )
        source_modes = torch.tensor([1, 1, 0])
        gate = source_mode_gate(logits, source_modes, tau=0.5)
        self.assertTrue(torch.equal(gate, torch.tensor([0.0, 1.0, 0.0])))

    def test_more_loss_freezes_classifier_but_keeps_feature_gradient(self) -> None:
        classifier = ModeClassifier(input_dim=3, num_modes=2, hidden_dim=8)
        loss_fn = MoreLoss(classifier, target_modes=(0,), gamma=0.2, tau=1.0)
        features = torch.randn(4, 3, requires_grad=True)
        retain_loss = torch.tensor(0.5, requires_grad=True)
        source_modes = torch.ones(4, dtype=torch.long)

        out = loss_fn(retain_loss, features, source_modes)
        out.loss.backward()

        self.assertIsNotNone(features.grad)
        self.assertTrue(all(param.grad is None for param in classifier.parameters()))

    def test_diffusion_policy_features_keep_model_gradient_path(self) -> None:
        class ToyScheduler:
            def __init__(self) -> None:
                self.config = SimpleNamespace(num_train_timesteps=4)
                self.alphas_cumprod = torch.tensor([0.9, 0.8, 0.7, 0.6])

            def add_noise(
                self,
                sample: torch.Tensor,
                noise: torch.Tensor,
                timesteps: torch.Tensor,
            ) -> torch.Tensor:
                alpha = self.alphas_cumprod[timesteps].view(sample.shape[0], 1, 1)
                return torch.sqrt(alpha) * sample + torch.sqrt(1.0 - alpha) * noise

        class ToyDenoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(5, 2)

            def forward(
                self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                global_cond: torch.Tensor,
            ) -> torch.Tensor:
                del timestep
                pred = self.linear(global_cond).unsqueeze(1)
                return pred.expand_as(sample)

        model = ToyDenoiser()
        scheduler = ToyScheduler()
        obs_condition = torch.randn(3, 5)
        action_chunks = torch.randn(3, 4, 2)

        out = diffusion_policy_features(
            denoising_model=model,
            noise_scheduler=scheduler,
            obs_condition=obs_condition,
            action_chunks=action_chunks,
            action_chunk_len=2,
        )

        self.assertEqual(out.features.shape, (3, 5 + 2 * 2))
        out.features.sum().backward()
        self.assertIsNotNone(model.linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
