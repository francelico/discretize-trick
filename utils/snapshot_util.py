from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from safetensors.torch import load_file

from utils.plot_util import build_color_lut, plot_node_ids_batched


def load_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    checkpoint = load_file(checkpoint_path)
    if checkpoint and next(iter(checkpoint)).startswith("_orig_mod."):
        checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model


def _latent_stats(normalization, latent: torch.Tensor):
    if normalization is None:
        return None, None
    mean = normalization["mean"].view(1, -1, 1, 1, 1).to(latent)
    std = normalization["std"].view(1, -1, 1, 1, 1).to(latent)
    return mean, std


@torch.no_grad()
def decode_voxel_latents(latents, decoder, normalization=None, batch_size=1):
    if isinstance(latents, list):
        latents = torch.stack(latents)
    batch, timesteps = latents.shape[:2]
    latents = rearrange(latents, "b t c x y z -> (b t) c x y z")
    mean, std = _latent_stats(normalization, latents)
    if mean is not None:
        latents = latents * std + mean

    voxel_classes = []
    decoder_device = next(decoder.parameters()).device
    for start in range(0, len(latents), batch_size):
        logits = decoder(latents[start : start + batch_size].to(decoder_device))
        voxel_classes.append(logits.argmax(dim=-1).cpu())
    voxel_classes = torch.cat(voxel_classes)
    return rearrange(voxel_classes, "(b t) x y z -> b t x y z", b=batch, t=timesteps)


@torch.no_grad()
def reencode_voxel_latents(latents, encoder, decoder, normalization=None, batch_size=1):
    batch, timesteps = latents.shape[:2]
    latents = rearrange(latents, "b t c x y z -> (b t) c x y z")
    mean, std = _latent_stats(normalization, latents)
    if mean is not None:
        latents = latents * std + mean

    reencoded = []
    decoder_device = next(decoder.parameters()).device
    for start in range(0, len(latents), batch_size):
        z = latents[start : start + batch_size].to(decoder_device)
        voxel_classes = decoder(z).argmax(dim=-1)
        reencoded.append(encoder(voxel_classes, sample_posterior=False))
    reencoded = torch.cat(reencoded)
    if mean is not None:
        reencoded = (reencoded - mean.to(reencoded)) / std.to(reencoded)
    return rearrange(reencoded, "(b t) c x y z -> b t c x y z", b=batch, t=timesteps)


def voxel_rollout_metrics(prediction: torch.Tensor, target: torch.Tensor):
    accuracy_by_time = (prediction == target).float().mean(dim=(-3, -2, -1))
    metrics = {
        "voxel_accuracy": accuracy_by_time.mean().item(),
        "voxel_accuracy_by_time": accuracy_by_time,
    }
    if prediction.shape[1] > 1:
        pred_changes = prediction[:, 1:] != prediction[:, :-1]
        target_changes = target[:, 1:] != target[:, :-1]
        change_agreement = (pred_changes == target_changes).float().mean(dim=(-3, -2, -1))
        metrics["change_agreement"] = change_agreement.mean().item()
        metrics["change_agreement_by_time"] = change_agreement
    return metrics


def render_voxel_classes(
    voxel_classes,
    voxel_classdict,
    node_registry_path,
    render_res=512,
    render_backend="sdl2",
):
    node_registry_path = Path(node_registry_path).expanduser()
    if not node_registry_path.is_file():
        raise FileNotFoundError(f"Node registry not found: {node_registry_path}")
    original_shape = voxel_classes.shape[:-3]
    voxel_classes = voxel_classes.detach().cpu().long().reshape(
        -1, *voxel_classes.shape[-3:]
    )
    grid_shape = voxel_classes.shape[-3:]
    coords = torch.stack(
        torch.meshgrid(
            *(torch.linspace(-0.5, 0.5, size) for size in grid_shape),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3).numpy()

    class_mapping = voxel_classdict["node_classes"]
    class_to_node = np.zeros((max(map(int, class_mapping)) + 1, 2), dtype=np.int32)
    for class_id, node_id_param in class_mapping.items():
        class_to_node[int(class_id)] = node_id_param
    flat_classes = voxel_classes.reshape(len(voxel_classes), -1).numpy()
    node_ids_params = class_to_node[flat_classes]

    node_registry = pd.read_csv(node_registry_path)
    images = plot_node_ids_batched(
        node_ids_params,
        coords,
        get_rgba=build_color_lut(node_registry),
        figsize=(render_res, render_res),
        backend=render_backend,
    )
    images = images[..., :3]
    return images.reshape(*original_shape, *images.shape[-3:])


def save_snapshot_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
