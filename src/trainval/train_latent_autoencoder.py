import argparse
import logging
import os
import time
from typing import Any, Dict, List, Tuple

from gaussian_renderer import render
from scene.cameras import MiniCam

from src.datasets import Scan3RObjectDataset
from src.representations.gaussian.gaussian_model import Gaussian
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov

# set cuda launch blocking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets.loaders import get_train_val_data_loader, get_val_dataloader
from src.engine import EpochBasedTrainer
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.losses.reconstruction import LPIPS
from utils import common, scan3r
from utils.gaussian_splatting import GaussianSplat
from utils.loss_utils import l1_loss, ssim

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)

        # Model Specific params
        self.root_dir = cfg.data.root_dir
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules

        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight

        # Dataloader
        start_time: float = time.time()

        train_loader, val_loader = get_train_val_data_loader(
            cfg, dataset=Scan3RObjectDataset
        )

        loading_time: float = time.time() - start_time
        message: str = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model
        model = self.create_model()
        self.register_model(model)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay,
            eps=1e-3,
        )
        self.register_optimizer(optimizer)

        # scheduler
        if cfg.train.optim.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                cfg.train.optim.lr_decay_steps,
                gamma=cfg.train.optim.lr_decay,
            )
        elif cfg.train.optim.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.train.optim.T_max,
                eta_min=cfg.train.optim.lr_min,
                T_mult=cfg.train.optim.T_mult,
                last_epoch=-1,
            )
        elif cfg.train.optim.scheduler == "linear":
            scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: (
                    1.0
                    if epoch <= cfg.train.optim.sched_start_epoch
                    else (
                        1.0
                        if epoch >= cfg.train.optim.sched_end_epoch
                        else (
                            1
                            - (epoch - cfg.train.optim.sched_start_epoch)
                            / (
                                cfg.train.optim.sched_end_epoch
                                - cfg.train.optim.sched_start_epoch
                            )
                        )
                        + (cfg.train.optim.end_lr / cfg.train.optim.lr)
                        * (epoch - cfg.train.optim.sched_start_epoch)
                        / (
                            cfg.train.optim.sched_end_epoch
                            - cfg.train.optim.sched_start_epoch
                        )
                    )
                ),
            )
        else:
            scheduler = None

        if scheduler is not None:
            self.register_scheduler(scheduler)

        self.logger.info("Initialisation Complete")

    def create_model(self) -> LatentAutoencoder:
        if self.cfg.autoencoder.guidance:
            from src.guidance.text_guidance import TextGuidance

            self.text_guidance = TextGuidance(device=self.device)
            model = LatentAutoencoder(
                cfg=self.cfg.autoencoder,
                device=self.device,
                text_guidance=self.text_guidance,
            )
        else:
            model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)

        if self.cfg.train.freeze_encoder:
            self.freeze_encoder()

        self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        return model

    def freeze_encoder(self) -> None:
        assert self.model is not None and isinstance(self.model, LatentAutoencoder)
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, LatentAutoencoder)

        img_poses = data_dict["scene_graphs"]["obj_img_poses"]
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        obj_ids = data_dict["scene_graphs"]["obj_ids"]
        intrinsic = data_dict["scene_graphs"]["obj_intrinsics"]
        frames = data_dict["scene_graphs"]["obj_img_top_frames"]
        masks = data_dict["scene_graphs"]["obj_annos"]
        translations = data_dict["scene_graphs"]["mean_obj_splat"]
        scales = data_dict["scene_graphs"]["scale_obj_splat"]

        embedding = self.model.encode(data_dict)
        reconstruction = self.model.decode(embedding)

        predicted_images = []
        ground_truth_images = []
        ground_truths = []

        for i in range(len(reconstruction)):
            scene_id = scene_ids[i][0]
            obj_id = obj_ids[i]
            intrinsics = intrinsic[scene_id]
            pose_idx = np.random.randint(0, len(img_poses[scene_id][obj_id]))
            frame_id = frames[scene_id][obj_id][pose_idx]
            extrinsics = img_poses[scene_id][obj_id][pose_idx]
            image = Image.open(
                f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{frame_id}.color.jpg"
            )
            image = (
                torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            )  # SHape: (3, H, W)
            mask = masks[scene_id][frame_id]
            mask = np.where(mask == int(obj_id), 1, 0)  # SHape: (H, W)
            image_masked = image * mask

            pose_camera_to_world = np.linalg.inv(pose_quatmat_to_rotmat(extrinsics))
            viewpoint_camera = MiniCam(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
                znear=0.01,
                zfar=100.0,
                R=pose_camera_to_world[:3, :3].T,
                T=pose_camera_to_world[:3, 3],
            )
            reconstruction[i].rescale(
                torch.tensor([2, 2, 2], device=reconstruction[i].get_xyz.device)
            )
            reconstruction[i].translate(
                -torch.tensor([1, 1, 1], device=reconstruction[i].get_xyz.device)
            )
            reconstruction[i].rescale(scales[i])
            reconstruction[i].translate(translations[i])

            rendered_image = render(
                viewpoint_camera,
                reconstruction[i],
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            predicted_images.append(rendered_image)
            ground_truth_images.append(image_masked)

        if len(predicted_images) == 0:
            return {}, {}

        predicted_images = torch.stack(predicted_images).squeeze(1)
        ground_truth_images = torch.stack(ground_truth_images).squeeze(1).cuda().float()
        protometric_loss = 0.8 * l1_loss(
            predicted_images, ground_truth_images
        ) + 0.2 * (1.0 - ssim(predicted_images, ground_truth_images))
        perceptual_loss = self.perceptual_loss(predicted_images, ground_truth_images)

        reconstruction: List[Gaussian]
        volume_loss = torch.tensor(
            [recon.get_scaling.prod(dim=-1).mean() for recon in reconstruction]
        ).mean()
        opacity_loss = torch.tensor(
            [((1 - recon.get_opacity) ** 2).mean() for recon in reconstruction]
        ).mean()

        loss = (
            protometric_loss + volume_loss + 0.001 * opacity_loss + perceptual_loss
        ) * self.cfg.train.loss.decoder_weight
        loss_dict = {
            "loss": loss * 100,
            "l1_loss": protometric_loss,
            "volume_loss": volume_loss,
            "opacity_loss": opacity_loss,
            "perceptual_loss": perceptual_loss,
        }
        output_dict = {
            "reconstruction": reconstruction,
            "gt": ground_truths,
            "predicted_images": predicted_images,
            "ground_truth_images": ground_truth_images,
            "embeddings": embedding,
        }
        return output_dict, loss_dict

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        self._save_embeddings(epoch, iteration, data_dict, output_dict)

    def _save_embeddings(self, epoch, iteration, data_dict, output_dict):
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        obj_ids = data_dict["scene_graphs"]["obj_ids"]
        embeddings = output_dict["embeddings"]
        os.makedirs(f"{self.cfg.output_dir}/embeddings", exist_ok=True)
        for i in range(embeddings.shape[0]):
            scene_id = scene_ids[i][0]
            obj_id = obj_ids[i]
            torch.save(
                embeddings[i],
                f"{self.cfg.output_dir}/embeddings/{scene_id}_{obj_id}.pt",
            )

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            return self.train_step(epoch, iteration, data_dict)

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        self._save_embeddings(epoch, iteration, data_dict, output_dict)

    def set_eval_mode(self) -> None:
        self.training = False
        self.model.eval()
        self.perceptual_loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self) -> None:
        self.training = True
        self.model.train()
        self.perceptual_loss.train()
        torch.set_grad_enabled(True)

    def visualize(
        self, output_dict: Dict[str, Any], epoch: int, mode: str = "train"
    ) -> None:
        predicted_images = output_dict["predicted_images"]
        ground_truth_images = output_dict["ground_truth_images"]
        reconstructions = output_dict["reconstruction"]

        for i in range(len(reconstructions[:4])):
            reconstructions[i].save_ply(
                f"{self.cfg.output_dir}/events/{mode}_reconstruction_{i}.ply"
            )

        save_image(
            predicted_images[:4],
            f"{self.cfg.output_dir}/events/{mode}_predicted_images.png",
        )
        save_image(
            ground_truth_images[:4],
            f"{self.cfg.output_dir}/events/{mode}_ground_truth_images.png",
        )

        side_by_side_images = torch.concat(
            [ground_truth_images, predicted_images],
            dim=-1,
        )
        self.writer.add_image(
            f"{mode}/reconstructions",
            side_by_side_images[:4],
            global_step=epoch,
            dataformats="NCHW",
        )


def parse_args(
    parser: argparse.ArgumentParser = None,
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--config", dest="config", default="", type=str, help="configuration name"
    )
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--snapshot", default=None, help="load from snapshot")
    parser.add_argument(
        "--load_encoder", default=None, help="name of pretrained encoder"
    )
    parser.add_argument("--epoch", type=int, default=None, help="load epoch")
    parser.add_argument("--log_steps", type=int, default=1, help="logging steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for ddp")

    args, unknown_args = parser.parse_known_args()
    return parser, args, unknown_args


def main() -> None:
    """Run training."""

    common.init_log(level=logging.INFO)
    parser, args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
