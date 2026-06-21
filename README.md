# Behavior Uncloning

This repository contains a minimal release of the code for:

**Behavior Uncloning: Distilling Mode Redirection into Policy Weights without Inference-Time Steering**

The paper introduces **MoRE** (Mode Redirection), a post-hoc policy-editing method
for redirecting a mixed-mode behavior-cloned robot policy toward desired behavior
modes. MoRE trains a temporary mode classifier, uses it to provide a
classifier-guided redirect loss during editing, preserves desired-mode behavior
with a retain loss, and then discards the classifier. The edited policy runs with
the original inference path.

The paper PDF will be added after public release. Draft submission PDFs are not
tracked in this repository.

## What is included

- `more/classifiers.py`: the mode-classifier MLP used by MoRE.
- `more/train_classifier.py`: a small helper for training the temporary
  classifier on cached original-policy features.
- `more/losses.py`: subset redirect loss, source-mode gate, and MoRE loss helpers.
- `more/diffusion_policy.py`: Diffusion Policy feature and retain-loss helpers
  matching the paper's `(condition, predicted clean action chunk)` classifier
  input.
- `more/trainer.py`: a compact trainer skeleton for plugging MoRE into a policy.
- `examples/minimal_more_edit.py`: a small synthetic example showing the expected
  policy-specific hooks. This is a smoke test, not a paper experiment.
- `tests/test_more_core.py`: focused tests for the method-defining MoRE losses
  and feature path.

This release intentionally avoids the internal experiment tree, private paths,
intermediate baselines, logs, checkpoints, and code-editing history from the
development repository.

This repository is a compact method release, not a full reproduction bundle for
every simulated and real-robot experiment in the paper.

## Installation

```sh
python -m pip install -r requirements.txt
```

For editable development installs:

```sh
python -m pip install -e .
```

## Minimal Example

```sh
python examples/minimal_more_edit.py
```

Run the lightweight tests with:

```sh
python -m unittest discover -s tests
```

For a real robot-policy integration, provide:

1. A mode-labeled editing dataset.
2. A frozen mode classifier trained on features from the original mixed policy.
3. A policy-specific `feature_fn(policy, batch)` that returns differentiable
   classifier features under the current policy.
4. A policy-specific `retain_loss_fn(policy, batch)` that computes the original
   behavior-cloning loss on desired-mode samples.

MoRE then optimizes:

```text
retain_loss(desired samples) + gamma * redirect_loss(gated undesired samples)
```

where redirect loss moves classifier probability mass toward the desired mode set
and the gate applies redirection only when the source-mode probability is below
`tau` (default `0.5` in the paper).

## Citation

Citation information will be added after release.

## Project Website

The static project website lives in `docs/` and is also pushed to the
`gh-pages` branch for GitHub Pages deployment.

To publish it on GitHub Pages, enable Pages in the repository settings and use:

- Source: `Deploy from a branch`
- Branch: `gh-pages`
- Folder: `/root`
