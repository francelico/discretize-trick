# AGENTS.md

Guidance for coding agents working in this repository.

## Scope

This is a deliberately narrow research codebase. It supports only:

- training and snapshotting the voxel VAE;
- training and autoregressive snapshotting the voxel DiT;
- optional VAE decode/re-encode after each generated DiT rollout step;
- local and Weights & Biases logging.

Do not add pixel models, end-to-end inference pipelines, dataset-generation scripts, Craftium,
Minetest environment bindings, or unrelated models unless the user explicitly expands the scope.
Datasets are assumed to be prebuilt and downloaded separately.

## Environment

- Use Python 3.11 and `uv`; install with `uv sync`.
- Run commands from the repository root.
- The default CUDA environment is PyTorch 2.7.1 with CUDA 12.6 wheels.
- Use `uv run python ...` or `uv run accelerate ...`; do not depend on globally installed packages.
- Snapshot rendering uses VisPy and `assets/node_registry.txt`. SDL2 is the local default; EGL is
  available through the CLI for suitably configured headless machines.

## Architecture

- `train_scripts/train_vae_voxel.py`: categorical voxel VAE training and reconstruction snapshots.
- `train_scripts/train_diffuser_voxel.py`: voxel-latent flow matching and autoregressive snapshots.
- `models/vae_voxel.py`: `ResNet3dEncoder` and `ResNet3dDecoder`.
- `models/dit_voxel.py`: `VoxelDiT`; shared attention and embedding code is in the adjacent modules.
- `trainers/__init__.py`: Accelerate loop, EMA, checkpoint resume/save, periodic snapshots, W&B.
- `trainers/flow_euler.py`: training diffusion helpers and tensor-only autoregressive Euler sampler.
- `data_loaders/`: read-only loaders for the prebuilt dataset described in `README.md`.
- `utils/snapshot_util.py`: checkpoint loading, latent decode/re-encode, metrics, rendering, NPZ save.
- `utils/plot_util.py`: the batched VisPy voxel renderer.

The re-encoding trick is implemented by passing `reencode_voxel_latents` into
`FlowEuler.denoise_autoreg`; it must run only after a newly generated timestep is fully denoised.
Model inputs and outputs in `denoise_autoreg` are tensors, not tuples.

## Working conventions

- Keep changes small and research-oriented. Prefer explicit code over abstraction for two callers.
- Preserve the prebuilt dataset schema and checkpoint compatibility unless a migration is requested.
- Do not silently copy code from a larger upstream repository. Port only a dependency proven to be
  required by one of the two supported workflows.
- Keep renderer-independent snapshot data in compressed NPZ files as well as visual media.
- Never embed W&B API keys. Authenticate with `uv run python -m wandb login` and enable online runs
  with `WANDB_MODE=online`, `--track-with-wandb`, `--wandb-entity`, and `--wandb-project-name`.
- Architecture CLI arguments must match resumed checkpoints. Dataset vocabulary size must match VAE
  input/output channels.

## Validation

For changes to shared code, at minimum check both entry points with `--help` and import all modules.
For behavior changes, use a tiny model and a small prebuilt dataset to exercise:

1. one VAE optimizer step and a VAE snapshot;
2. one DiT optimizer step;
3. a short DiT autoregressive snapshot with re-encoding enabled.

Do not commit generated datasets, checkpoints, `outputs/`, `.venv/`, or W&B run directories.
