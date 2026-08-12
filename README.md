# discretize-trick

A small research repository for training a voxel VAE and a voxel latent diffusion transformer,
then inspecting both models with local or Weights & Biases snapshots. The DiT snapshot supports
the discretization/re-encoding trick: every predicted latent can be decoded to a voxel-class grid
and deterministically encoded again before the next autoregressive step.

This repository intentionally excludes pixel-model training, inference pipelines, dataset-building
tools, and game environments. It expects a prebuilt dataset.

## Setup

Requirements:

- Python 3.11
- CUDA 12.6-compatible NVIDIA driver for the pinned PyTorch build
- SDL2 for the default snapshot renderer
- `uv`

Install the environment:

```bash
uv sync
```

Run scripts from the repository root so local package imports and `assets/node_registry.txt`
resolve correctly. Inspect all architecture and training flags with `--help`:

```bash
uv run python train_scripts/train_vae_voxel.py --help
uv run python train_scripts/train_diffuser_voxel.py --help
```

## Prebuilt dataset

Point `--dataset-path` at a directory with this layout:

```text
dataset/
├── metadata.csv
├── dataset_params.json
├── mt_voxel_classdict.json
├── latent_stats.npz
├── voxel_classes/<instance>.npz
├── voxel_latents/<instance>.safetensors
├── pixel_latents/<instance>.safetensors
└── <raw episode path from metadata.csv>/data.npz
```

The VAE needs `metadata.csv`, `mt_voxel_classdict.json`, and `voxel_classes`. The DiT also needs
the voxel and pixel latent folders, raw episode `data.npz` files for actions/cameras, and
`latent_stats.npz` when latent normalization is enabled. Metadata must contain `sha256`,
`validation_set`, `level_quality_score`, and the generated-data status columns used by the loaders.
The DiT snapshot additionally reads voxel classes. Importance masks are required when
`--remove-bad-frames` is enabled; use `--no-remove-bad-frames` if the downloaded dataset omits them.

Model dimensions must match the dataset. In particular, VAE input/output channels must equal the
number of `node_classes` in `mt_voxel_classdict.json`, and the DiT latent and conditioning shapes
must match the stored latents.

## Train

Single-GPU voxel VAE:

```bash
uv run accelerate launch train_scripts/train_vae_voxel.py \
  --run-name voxel-vae \
  --dataset-path datasets/mydata \
  --total-grad-steps 100000 \
  <encoder/decoder architecture args>
```

Single-GPU voxel DiT:

```bash
uv run accelerate launch train_scripts/train_diffuser_voxel.py \
  --run-name voxel-dit \
  --dataset-path datasets/mydata \
  --total-grad-steps 100000 \
  <denoiser architecture args> \
  diffusion:diffusion-forcing
```

Use `accelerate launch --multi_gpu --num_processes <N> ...` for multi-GPU training. Runs resume
automatically from the newest matching run under `--output-dir`; pass `--resume disable` for a new
run or `--resume <checkpoint_dir>` for an explicit Accelerator checkpoint.

## Snapshots

The bundled node registry supplies colors for the 3D voxel renderer. SDL2 is the default backend;
EGL can be selected with the corresponding render-backend flag on a headless system.

Snapshot an existing voxel VAE checkpoint:

```bash
uv run python train_scripts/train_vae_voxel.py \
  --snapshot-only \
  --resume outputs/voxel-vae/<run>/checkpoints/checkpoint_<step> \
  --dataset-path datasets/mydata \
  <matching encoder/decoder architecture args>
```

Snapshot an existing voxel DiT with re-encoding enabled:

```bash
uv run python train_scripts/train_diffuser_voxel.py \
  --snapshot-only \
  --resume outputs/voxel-dit/<run>/checkpoints/checkpoint_<step> \
  --dataset-path datasets/mydata \
  --snapshot.decoder-checkpoint <vae_decoder.safetensors> \
  --snapshot.encoder-checkpoint <vae_encoder.safetensors> \
  --snapshot.reencode-latents \
  --snapshot.autoreg-num-seq 2 \
  --snapshot.autoreg-seq-len 16 \
  --snapshot.autoreg-num-diffusion-steps 20 \
  <matching denoiser and VAE architecture args> \
  diffusion:diffusion-forcing
```

Use `--i-snapshot <grad_steps>` during normal training for periodic snapshots. VAE snapshots save
NPZ metrics and side-by-side reconstruction PNGs. DiT snapshots save NPZ metrics and side-by-side
ground-truth/autoregressive MP4s under `<run>/snapshots/`.

## Weights & Biases

Authenticate once, then add the shared logging flags to either training or snapshot command:

```bash
uv run python -m wandb login

WANDB_MODE=online uv run python train_scripts/<script>.py \
  --track-with-wandb \
  --wandb-entity <entity> \
  --wandb-project-name <project> \
  <args>
```

Training metrics use `trainer/grad_step` as their W&B step. VAE snapshots upload reconstruction
images, cross-entropy, and voxel accuracy. DiT snapshots upload rollout videos, latent MSE, voxel
accuracy, and change agreement.

## Repository layout

```text
train_scripts/   VAE and DiT entry points
models/          voxel VAE, voxel DiT, and shared neural-network blocks
trainers/        training loop, checkpointing, EMA, W&B, and FlowEuler
data_loaders/    readers for an already-built voxel/latent dataset
utils/           training, camera, snapshot, and renderer helpers
assets/          voxel node color registry used by snapshots
```
