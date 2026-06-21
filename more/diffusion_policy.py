from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DiffusionFeatureOutput:
    features: torch.Tensor
    noise_pred: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    predicted_clean_actions: torch.Tensor


def diffusion_policy_features(
    denoising_model: nn.Module,
    noise_scheduler: object,
    obs_condition: torch.Tensor,
    action_chunks: torch.Tensor,
    *,
    feature_condition: torch.Tensor | None = None,
    action_chunk_len: int | None = None,
) -> DiffusionFeatureOutput:
    """Build MoRE classifier features for a Diffusion Policy-style model.

    The paper uses r_theta(x) = (c_t, ahat_0,theta), where c_t is the policy
    condition and ahat_0,theta is the predicted clean action chunk reconstructed
    from the denoising output. This function keeps the autograd path from the
    classifier features back into the denoising model.

    `noise_scheduler` is expected to expose the Diffusers-style methods/fields:
    `add_noise`, `config.num_train_timesteps`, and `alphas_cumprod`.
    """

    device = action_chunks.device
    batch_size = action_chunks.shape[0]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(action_chunks)
    noisy_actions = noise_scheduler.add_noise(action_chunks, noise, timesteps)
    noise_pred = denoising_model(
        sample=noisy_actions,
        timestep=timesteps,
        global_cond=obs_condition,
    )

    alpha_bar = noise_scheduler.alphas_cumprod.to(device)[timesteps].view(batch_size, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    predicted_clean = (noisy_actions - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar

    chunk = predicted_clean
    if action_chunk_len is not None:
        chunk = chunk[:, :action_chunk_len]
    condition = obs_condition if feature_condition is None else feature_condition
    features = torch.cat([condition, chunk.flatten(start_dim=1)], dim=-1)
    return DiffusionFeatureOutput(
        features=features,
        noise_pred=noise_pred,
        noise=noise,
        timesteps=timesteps,
        predicted_clean_actions=predicted_clean,
    )


def diffusion_policy_retain_loss(
    denoising_model: nn.Module,
    noise_scheduler: object,
    obs_condition: torch.Tensor,
    action_chunks: torch.Tensor,
) -> torch.Tensor:
    """Standard Diffusion Policy noise-prediction loss for desired-mode retain data."""

    out = diffusion_policy_features(
        denoising_model=denoising_model,
        noise_scheduler=noise_scheduler,
        obs_condition=obs_condition,
        action_chunks=action_chunks,
    )
    return F.mse_loss(out.noise_pred, out.noise)
