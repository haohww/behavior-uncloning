from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class ClassifierTrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"


def train_mode_classifier(
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: ClassifierTrainConfig | None = None,
) -> dict[str, float]:
    """Train a K-way mode classifier on cached original-policy features.

    Each loader batch must be either `(features, labels)` or a dict containing
    `features` and `labels`. Features should be computed with the original mixed
    policy, as described in the paper.
    """

    cfg = config or ClassifierTrainConfig()
    device = torch.device(cfg.device)
    classifier.to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    last_train_loss = float("nan")
    for _ in range(cfg.epochs):
        classifier.train()
        losses = []
        for batch in train_loader:
            features, labels = _unpack_batch(batch, device)
            logits = classifier(features)
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=cfg.label_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if losses:
            last_train_loss = sum(losses) / len(losses)

    metrics = {"train_loss": last_train_loss}
    if val_loader is not None:
        metrics.update(evaluate_mode_classifier(classifier, val_loader, device))
    return metrics


@torch.no_grad()
def evaluate_mode_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    device = torch.device(device or next(classifier.parameters()).device)
    classifier.eval().to(device)
    losses = []
    correct = 0
    total = 0
    for batch in loader:
        features, labels = _unpack_batch(batch, device)
        logits = classifier(features)
        losses.append(float(F.cross_entropy(logits, labels)))
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum())
        total += int(labels.numel())
    return {
        "val_loss": sum(losses) / len(losses) if losses else float("nan"),
        "val_acc": correct / total if total else float("nan"),
    }


def _unpack_batch(
    batch: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        features = batch["features"]
        labels = batch["labels"]
    else:
        features, labels = batch  # type: ignore[misc]
    return features.to(device), labels.to(device, dtype=torch.long)
