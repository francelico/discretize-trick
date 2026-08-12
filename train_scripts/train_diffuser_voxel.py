import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration
from easydict import EasyDict as edict
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader

from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraAction
from models import dit_voxel, vae_voxel
from trainers import BaseTrainer
from trainers.flow_euler import FlowEuler, FlowEulerTrainingArgs
from utils.snapshot_util import (
    decode_voxel_latents,
    load_weights,
    reencode_voxel_latents,
    render_voxel_classes,
    save_snapshot_npz,
    voxel_rollout_metrics,
)
from utils.train_util import infinite_loader

warnings.simplefilter(action="ignore", category=FutureWarning)


def fast_psnr(mse, data_range, base=10.0):
    """Compute PSNR from MSE and data range."""
    psnr_base_e = 2 * torch.log(torch.tensor(data_range)) - torch.log(torch.maximum(mse, torch.tensor(1e-10)))
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr


@dataclass
class SnapshotArgs:
    decoder: vae_voxel.ResNetDecoderArgs = field(default_factory=vae_voxel.ResNetDecoderArgs)
    encoder: vae_voxel.ResNetEncoderArgs = field(default_factory=vae_voxel.ResNetEncoderArgs)
    decoder_checkpoint: str | None = None
    encoder_checkpoint: str | None = None
    decoder_batch_size: int = 1
    autoreg_batch_size: int = 1
    autoreg_seq_len: int = 16
    autoreg_num_seq: int = 2
    autoreg_num_diffusion_steps: int = 20
    num_context_frames: int = 1
    reencode_latents: bool = False
    instances: str | None = None
    tag: str = ""
    node_registry_path: str = "assets/node_registry.txt"
    render_backend: Literal["egl", "sdl2"] = "sdl2"
    render_res: int = 512


@dataclass(kw_only=True)
class TrainingArgs(FlowEulerTrainingArgs):
    """Configuration for training the model."""

    denoiser: dit_voxel.VoxelDitDenoiserArgs
    """Denoiser model to use for diffusion."""

    run_name: str = "debug-diffuser-voxel"
    """Experiment name"""

    output_dir: str = "outputs/runs/voxel_diffuser"
    """Output directory"""

    loss_target: Literal["x0", "v"] = "v"
    """Loss target for diffusion model"""

    camera_offset_ts: int = 1
    """Time offset to apply to camera conditioning"""

    camera_translation_representation: Literal["extrinsics", "voxel_local"] = "voxel_local"

    camera_rotation_representation: Literal["quaternion", "6d", "cam_dir"] = '6d'

    remove_bad_frames: bool = True
    """Whether to filter out frames not satisfying delay/inconsistency heuristics."""

    snapshot: SnapshotArgs = field(default_factory=SnapshotArgs)

    snapshot_only: bool = False

    i_snapshot: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.snapshot.decoder.latent_channels = self.denoiser.in_channels
        self.snapshot.encoder.latent_channels = self.denoiser.in_channels
        if self.snapshot_only or self.i_snapshot > 0:
            if self.snapshot.decoder_checkpoint is None:
                raise ValueError("A snapshot decoder checkpoint is required when snapshotting is enabled.")
            if self.snapshot.reencode_latents and self.snapshot.encoder_checkpoint is None:
                raise ValueError("An encoder checkpoint is required when re-encoding latents.")
            if self.snapshot.autoreg_num_seq < 1 or self.snapshot.autoreg_batch_size < 1:
                raise ValueError("Snapshot sequence and batch counts must be positive.")
            if self.snapshot.autoreg_seq_len < 1:
                raise ValueError("snapshot.autoreg_seq_len must be positive.")
            if not 0 <= self.snapshot.num_context_frames < self.denoiser.context_window_size:
                raise ValueError("snapshot.num_context_frames must be smaller than the denoiser context window.")
        #raymap currently only supports voxel_local
        if self.denoiser.pixel_use_raymap:
            assert self.camera_translation_representation == "voxel_local"


class VoxelDiffuserTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)
        self.flow_euler = FlowEuler(cfg.diffusion, device=accelerator.device)

    def prepare_dataset(self):
        logger.info("Build Datasets ...")
        self.dataset = MinetestLatentCameraAction(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
            clip_len=self.cfg.denoiser.context_window_size,
            normalize_latents=self.cfg.normalize_latents,
            pixel_offset_ts=1,
            camera_offset_ts=self.cfg.camera_offset_ts,
            cam_rotation_representation=self.cfg.camera_rotation_representation,
            cam_translation_representation=self.cfg.camera_translation_representation,
            sample_voxel_importance_mask=self.cfg.remove_bad_frames,
            remove_bad_frames=self.cfg.remove_bad_frames,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader_workers,
            persistent_workers=True if self.cfg.dataloader_workers else False,
            prefetch_factor=self.cfg.dataloader_prefetch_factor if self.cfg.dataloader_workers else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=getattr(self.dataset, "collate_fn", None),
        )

        self.val_dataset = MinetestLatentCameraAction(
            self.cfg.dataset_path,
            min_level_quality_score=self.cfg.min_level_quality_score,
            clip_len=self.cfg.denoiser.context_window_size,
            split="val",
            normalize_latents=self.cfg.normalize_latents,
            pixel_offset_ts=1,
            camera_offset_ts=self.cfg.camera_offset_ts,
            cam_rotation_representation=self.cfg.camera_rotation_representation,
            cam_translation_representation=self.cfg.camera_translation_representation,
            sample_voxel_importance_mask=self.cfg.remove_bad_frames,
            remove_bad_frames=self.cfg.remove_bad_frames,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.val_dataloader_workers,
            persistent_workers=True if self.cfg.val_dataloader_workers else False,
            prefetch_factor=self.cfg.val_dataloader_prefetch_factor if self.cfg.val_dataloader_workers else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        if self.cfg.snapshot_only or self.cfg.i_snapshot > 0:
            snapshot_clip_len = (
                self.cfg.snapshot.num_context_frames + self.cfg.snapshot.autoreg_seq_len
            )
            snapshot_dataset_kwargs = dict(
                root=self.cfg.dataset_path,
                clip_len=snapshot_clip_len,
                sample_from_start_of_episode=True,
                min_level_quality_score=self.cfg.min_level_quality_score,
                instances=self.cfg.snapshot.instances,
                normalize_latents=self.cfg.normalize_latents,
                sample_voxel_classes=True,
                pixel_offset_ts=1,
                camera_offset_ts=self.cfg.camera_offset_ts,
                cam_rotation_representation=self.cfg.camera_rotation_representation,
                cam_translation_representation=self.cfg.camera_translation_representation,
            )
            self.snapshot_train_dataset = MinetestLatentCameraAction(
                split="train", **snapshot_dataset_kwargs
            )
            self.snapshot_val_dataset = MinetestLatentCameraAction(
                split="val", **snapshot_dataset_kwargs
            )
            if not len(self.snapshot_train_dataset) or not len(self.snapshot_val_dataset):
                raise ValueError("Snapshot train and validation splits must both contain episodes.")
            loader_kwargs = dict(
                batch_size=self.cfg.snapshot.autoreg_batch_size,
                shuffle=False,
                num_workers=self.cfg.dataloader_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=self.snapshot_train_dataset.collate_fn,
            )
            self.snapshot_train_dataloader = DataLoader(
                self.snapshot_train_dataset, **loader_kwargs
            )
            loader_kwargs["collate_fn"] = self.snapshot_val_dataset.collate_fn
            self.snapshot_val_dataloader = DataLoader(
                self.snapshot_val_dataset, **loader_kwargs
            )

    # fmt:off
    def prepare_models(self):
        logger.info("Build models ...")
        self.models = nn.ModuleDict(
            {
                "denoiser": getattr(dit_voxel, self.cfg.denoiser.name)(**vars(self.cfg.denoiser)
                ),
            }
        )
        num_parameters = sum(p.numel() for p in self.models.parameters())
        logger.info(f"# of parameters: {num_parameters / 1e6: .4f}M")

    def training_losses(
            self,
            batch,
            verbose=True,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            batch: the batch of data

        Returns:
            a dict with the key "loss" containing a tensor of shape [B].
            may also contain other keys for different terms.
        """
        x_0 = batch["voxel"]

        # build conditioning
        cond = {
            "action": batch["action"].clone(),
            "camera": batch["camera"].clone(),
        }
        # noise augmentation on pixel latents
        noise_pix = torch.randn_like(batch["pixel"])
        noise_pix = torch.clamp(noise_pix, -self.flow_euler.cfg.noise_abs_max, self.flow_euler.cfg.noise_abs_max)
        cond["pixel_latents"] = (1 - self.flow_euler.cfg.noise_cond) * batch["pixel"] + self.flow_euler.cfg.noise_cond * noise_pix

        noise = torch.randn_like(x_0)
        noise = torch.clamp(noise, -self.flow_euler.cfg.noise_abs_max, self.flow_euler.cfg.noise_abs_max)
        t = self.flow_euler.sample_t(*x_0.shape[:2]).to(x_0.device).float()
        x_t = self.flow_euler.diffuse(x_0, t, noise)

        pred = self.models['denoiser'](x_t, t, cond)
        if self.cfg.loss_target == "x0":
            target = x_0
        elif self.cfg.loss_target == "v":
            target = self.flow_euler.get_v_from_noise(x_0, noise)
        else:
            raise ValueError(f"Unknown loss target: {self.cfg.loss_target}")

        terms = edict(loss=0.0)
        terms.loss = F.mse_loss(pred, target)
        if verbose:
            with torch.no_grad():
                loss_per_timestep = F.mse_loss(
                    pred, target, reduction="none").mean(dim=list(range(2, len(pred.shape))))

        if verbose:
            with torch.no_grad():
                if self.cfg.loss_target == "x0":
                    loss_data_space = F.mse_loss(pred, x_0).mean()
                elif self.cfg.loss_target == "v":
                    pred_x_0= self.flow_euler.sample_x0(x_t, t, pred)
                    loss_data_space = F.mse_loss(pred_x_0, x_0).mean()
                terms["z_psnr"] = fast_psnr(loss_data_space, data_range=2*self.flow_euler.cfg.noise_abs_max)
                time_bin = np.digitize(t.view(-1).cpu().numpy(), np.linspace(0, 1, 11)) - 1
                for i in range(10):
                    if (time_bin == i).sum() != 0:
                        terms[f"bin_{i}"] = loss_per_timestep.view(-1)[time_bin == i].mean()
                if "action" and "camera" in cond:
                    tt = rearrange(t, "B T -> (B T)")
                    t_emb = self.get_model(self.models['denoiser']).t_embedder(tt * self.cfg.denoiser.timestep_scaling_factor).abs()
                    t_emb = rearrange(t_emb, "(B T) D -> B T D", B=t.shape[0], T=t.shape[1])
                    camera_for_ada = cond["camera"].clone()
                    if self.get_model(self.models["denoiser"]).camera_use_ape:
                        camera_for_ada = rearrange(camera_for_ada, "b t d -> (b t) d")
                        cam_rot, cam_xyz, cam_fov = camera_for_ada.split([6, 3, 1], dim=-1)
                        cam_rot = self.get_model(self.models["denoiser"]).cam_embedding_layer_rot(cam_rot)
                        cam_xyz = self.get_model(self.models["denoiser"]).cam_embedding_layer_xyz(cam_xyz)
                        camera_for_ada = torch.cat((cam_rot, cam_xyz, cam_fov), dim=-1)
                        camera_for_ada = rearrange(camera_for_ada, "(b t) d -> b t d", b=t.shape[0], t=t.shape[1])
                    else:
                        camera_for_ada = camera_for_ada * self.get_model(self.models["denoiser"]).camera_pos_scaling_factor
                    if self.get_model(self.models["denoiser"]).action_camera_embedder:
                        act_emb = self.get_model(self.models["denoiser"]).action_camera_embedder(cond["action"], camera_for_ada).abs()
                        cam_emb = act_emb
                        c_emb = act_emb + t_emb
                    else:
                        act_emb = self.get_model(self.models["denoiser"]).action_embedder(cond["action"]).abs()
                        cam_emb = self.get_model(self.models["denoiser"]).camera_embedder(camera_for_ada).abs()
                        c_emb = act_emb + cam_emb + t_emb
                    terms["t_emb_frac"] = (t_emb / c_emb).mean()
                    terms["act_emb_frac"] = (act_emb / c_emb).mean()
                    terms["cam_emb_frac"] = (cam_emb / c_emb).mean()
                    terms["max_cam_pos"] = cond["camera"][..., -4:-1].abs().max()

        return terms, {}

    def load_snapshot_vae(self):
        with open(Path(self.cfg.dataset_path) / "mt_voxel_classdict.json") as file:
            self.voxel_classdict = json.load(file)
        vocab_size = len(self.voxel_classdict["node_classes"])
        if self.cfg.snapshot.decoder.out_channels != vocab_size:
            raise ValueError(
                f"Snapshot decoder has {self.cfg.snapshot.decoder.out_channels} output channels, "
                f"but the dataset has {vocab_size} voxel classes."
            )
        decoder = getattr(vae_voxel, self.cfg.snapshot.decoder.name)(
            **vars(self.cfg.snapshot.decoder)
        )
        load_weights(decoder, self.cfg.snapshot.decoder_checkpoint)
        decoder.eval().requires_grad_(False).to(self.accelerator.device)

        encoder = None
        if self.cfg.snapshot.reencode_latents:
            if self.cfg.snapshot.encoder.in_channels != vocab_size:
                raise ValueError(
                    f"Snapshot encoder has {self.cfg.snapshot.encoder.in_channels} input channels, "
                    f"but the dataset has {vocab_size} voxel classes."
                )
            encoder = getattr(vae_voxel, self.cfg.snapshot.encoder.name)(
                **vars(self.cfg.snapshot.encoder)
            )
            load_weights(encoder, self.cfg.snapshot.encoder_checkpoint)
            encoder.eval().requires_grad_(False).to(self.accelerator.device)
        return encoder, decoder

    @torch.no_grad()
    def run_autoreg(self, data_iterator, denoiser, encoder, decoder, normalization):
        snapshot_cfg = self.cfg.snapshot
        episode_length = snapshot_cfg.num_context_frames + snapshot_cfg.autoreg_seq_len
        reencode_fn = None
        if snapshot_cfg.reencode_latents:
            reencode_fn = partial(
                reencode_voxel_latents,
                encoder=encoder,
                decoder=decoder,
                normalization=normalization,
                batch_size=snapshot_cfg.decoder_batch_size,
            )

        result = {
            "instance_ids": [],
            "target_latents": [],
            "target_voxels": [],
            "from_context_latents": [],
        }
        num_generated = 0
        while num_generated < snapshot_cfg.autoreg_num_seq:
            episode = next(data_iterator)
            take = min(len(episode["voxel"]), snapshot_cfg.autoreg_num_seq - num_generated)
            episode = {
                key: value[:take] if isinstance(value, torch.Tensor) else value
                for key, value in episode.items()
            }
            voxel_latents = episode["voxel"][:, :episode_length]
            batch = len(voxel_latents)
            cond = {
                "action": episode["action"][:, :episode_length].to(self.accelerator.device),
                "camera": episode["camera"][:, :episode_length].to(self.accelerator.device),
                "pixel_latents": episode["pixel"][:, :episode_length].to(self.accelerator.device),
            }
            pixel_noise = torch.randn_like(cond["pixel_latents"]).clamp(
                -self.flow_euler.cfg.noise_abs_max,
                self.flow_euler.cfg.noise_abs_max,
            )
            cond["pixel_latents"] = (
                (1 - self.flow_euler.cfg.noise_cond) * cond["pixel_latents"]
                + self.flow_euler.cfg.noise_cond * pixel_noise
            )

            from_context = self.flow_euler.denoise_autoreg(
                denoiser,
                ep_length=episode_length,
                seq_length=self.cfg.denoiser.context_window_size,
                diffusion_steps=snapshot_cfg.autoreg_num_diffusion_steps,
                stable_t=self.flow_euler.cfg.min_noise_level_inference,
                cond=cond,
                x_start=voxel_latents[:, :snapshot_cfg.num_context_frames].to(
                    self.accelerator.device
                ),
                denoiser_pred=self.cfg.loss_target,
                reencode_fn=reencode_fn,
                verbose=False,
            ).sample

            result["instance_ids"].append(episode["instance_idx"].cpu())
            result["target_latents"].append(voxel_latents.cpu())
            result["target_voxels"].append(episode["voxel_classes"][:, :episode_length].cpu())
            result["from_context_latents"].append(from_context)
            num_generated += batch

        result = {key: torch.cat(value) for key, value in result.items()}
        result["from_context_voxels"] = decode_voxel_latents(
            result["from_context_latents"],
            decoder,
            normalization,
            snapshot_cfg.decoder_batch_size,
        )

        context = snapshot_cfg.num_context_frames
        result["from_context_latent_mse"] = F.mse_loss(
            result["from_context_latents"][:, context:].float(),
            result["target_latents"][:, context:].float(),
            reduction="none",
        ).mean(dim=(-4, -3, -2, -1))
        result["from_context_metrics"] = voxel_rollout_metrics(
            result["from_context_voxels"][:, context:],
            result["target_voxels"][:, context:],
        )
        return result

    @torch.no_grad()
    def snapshot(self):
        denoiser = self.accelerator.unwrap_model(self.models["denoiser"]).eval()
        gc.collect()
        torch.cuda.empty_cache()
        encoder, decoder = self.load_snapshot_vae()

        with self.accelerator.autocast():
            snapshots = {
                "train": self.run_autoreg(
                    infinite_loader(self.snapshot_train_dataloader),
                    denoiser,
                    encoder,
                    decoder,
                    self.snapshot_train_dataset.normalization["voxel_latent"],
                ),
                "val": self.run_autoreg(
                    infinite_loader(self.snapshot_val_dataloader),
                    denoiser,
                    encoder,
                    decoder,
                    self.snapshot_val_dataset.normalization["voxel_latent"],
                ),
            }

        tag = f"_{self.cfg.snapshot.tag}" if self.cfg.snapshot.tag else ""
        output_dir = Path(self.cfg.output_dir) / "snapshots" / f"voxel_step_{self.step}{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        wandb_values = {}
        tensor_keys = (
            "instance_ids",
            "target_voxels",
            "from_context_voxels",
            "from_context_latent_mse",
        )
        for split, result in snapshots.items():
            save_snapshot_npz(
                output_dir / f"{split}.npz",
                **{key: result[key] for key in tensor_keys},
            )
            metrics = result["from_context_metrics"]
            wandb_values[f"snapshot/{split}/from_context/latent_mse"] = result[
                "from_context_latent_mse"
            ].mean().item()
            wandb_values[f"snapshot/{split}/from_context/voxel_accuracy"] = metrics[
                "voxel_accuracy"
            ]
            if "change_agreement" in metrics:
                wandb_values[f"snapshot/{split}/from_context/change_agreement"] = metrics[
                    "change_agreement"
                ]

            render_kwargs = dict(
                voxel_classdict=self.voxel_classdict,
                node_registry_path=self.cfg.snapshot.node_registry_path,
                render_res=self.cfg.snapshot.render_res,
                render_backend=self.cfg.snapshot.render_backend,
            )
            target_frames = render_voxel_classes(result["target_voxels"], **render_kwargs)
            context_frames = render_voxel_classes(
                result["from_context_voxels"], **render_kwargs
            )
            panels = np.concatenate([target_frames, context_frames], axis=2)
            video_paths = []
            for index, panel in enumerate(panels):
                video_path = output_dir / f"{split}_{index}.mp4"
                iio.imwrite(video_path, panel, fps=4)
                video_paths.append(video_path)

            if self.wandb_run is not None:
                import wandb

                wandb_values[f"snapshot/{split}/rollouts"] = [
                    wandb.Video(
                        str(video_path),
                        format="mp4",
                        caption="ground truth / from context",
                    )
                    for video_path in video_paths
                ]
        self.log_wandb(wandb_values)

        denoiser.train()
        del encoder, decoder, snapshots
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Snapshot saved to {output_dir}")


def main(args):
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        automatic_checkpoint_naming=False,
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        project_config=project_config,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=10))],
        # step_scheduler_with_optimizer=False # uncomment this if you want to use an lr scheduler
    )
    trainer = VoxelDiffuserTrainer(args, accelerator)
    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()
    trainer.prepare_accelerate()
    if args.snapshot_only:
        if trainer.accelerator.is_main_process:
            trainer.snapshot()
            trainer.finish_logging()
        return
    trainer.train()


if __name__ == "__main__":
    # Prepare config
    args = tyro.cli(TrainingArgs)
    main(args)
