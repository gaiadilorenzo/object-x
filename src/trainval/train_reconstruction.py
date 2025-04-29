import argparse
import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from gaussian_renderer import render
from PIL import Image
from scene.cameras import MiniCam
from torch import nn
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets import Scan3RPatchObjectDataset
from src.datasets.loaders import get_train_val_data_loader
from src.engine import EpochBasedTrainer
from src.models.autoencoder import AutoEncoder
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.losses.reconstruction import mse_mask_loss
from src.modules.sparse.basic import SparseTensor
from utils import common
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


_FILL = 0.0


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)

        # Model Specific params
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules

        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight
        self.threshold = 0.5
        # Dataloader
        start_time: float = time.time()

        train_loader, val_loader = get_train_val_data_loader(
            cfg, dataset=Scan3RPatchObjectDataset
        )

        loading_time: float = time.time() - start_time
        message: str = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model
        model = self.create_model()
        self.register_model(model)

        self.loss = nn.MSELoss()
        self.masked_loss = mse_mask_loss

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

    def create_model(self) -> AutoEncoder:
        model = AutoEncoder(cfg=self.cfg.autoencoder, device=self.device)
        self.latent_autoencoder = LatentAutoencoder(
            cfg=self.cfg.autoencoder, device=self.device
        )
        self.latent_autoencoder.load_state_dict(
            torch.load(
                self.cfg.autoencoder.encoder.voxel.pretrained, map_location=self.device
            )["model"]
        )
        self.latent_autoencoder.eval()
        self.latent_autoencoder.to(self.device)
        for param in self.latent_autoencoder.parameters():
            param.requires_grad = False

        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        return model

    def compute_decaying_weight(
        self,
        initial_weight: float = 100.0,
        final_weight: float = 1.0,
        max_epochs: int = 5,
    ) -> float:
        """
        Compute an exponentially decaying weight.

        Args:
            epoch (int): Current epoch number.
            initial_weight (float): The starting weight value.
            final_weight (float): The weight value to decay towards.
            max_epochs (int): The epoch number at which the weight should approach final_weight.

        Returns:
            float: The decayed weight value for the given epoch.
        """
        decay_rate = -np.log(final_weight / initial_weight) / max_epochs
        weight = final_weight + (initial_weight - final_weight) * np.exp(
            -decay_rate * self.epoch
        )
        return weight

    def _densify(self, sparse_splat):
        sparse_splat_dense = sparse_splat.dense()
        dense_splat = torch.full(
            (
                sparse_splat_dense.shape[0],
                1,
                sparse_splat_dense.shape[2],
                sparse_splat_dense.shape[3],
                sparse_splat_dense.shape[4],
            ),
            _FILL,
            device=sparse_splat.device,
        )
        dense_splat[
            sparse_splat.coords[:, 0],
            :,
            sparse_splat.coords[:, 1],
            sparse_splat.coords[:, 2],
            sparse_splat.coords[:, 3],
        ] = 1.0
        return torch.cat((sparse_splat_dense, dense_splat), dim=1)

    def _sparsify(self, dense_splat):
        mask = dense_splat[:, -1] > self.threshold
        coords = torch.nonzero(mask, as_tuple=False)

        if len(coords) == 0:
            return None

        feats = dense_splat[coords[:, 0], :-1, coords[:, 1], coords[:, 2], coords[:, 3]]
        return SparseTensor(coords=coords.int(), feats=feats)

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, AutoEncoder)

        if data_dict["scene_graphs"]["tot_obj_splat"].shape[0] > 100:
            return {}, {}
        if not self.cfg.data.preload_slat:
            sparse_splat = self.latent_autoencoder.encode(data_dict)
        else:
            sparse_splat = data_dict["scene_graphs"]["tot_obj_splat"]

        # AVOID OVERFLOW CUDA
        if data_dict["scene_graphs"]["tot_obj_splat"].shape[0] > 100:
            self.logger.info(
                f"Skipping too big of a scene - {data_dict['scene_graphs']['tot_obj_splat'].shape[0]}"
            )
            return {}, {}

        if not self.cfg.data.preload_slat:
            sparse_splat = self.latent_autoencoder.encode(data_dict)
        else:
            sparse_splat = data_dict["scene_graphs"]["tot_obj_splat"]

        data_dict["scene_graphs"]["tot_obj_dense_splat"] = self._densify(sparse_splat)
        embedding = self.model.encode(data_dict)
        reconstruction = self.model.decode(embedding)

        mask = data_dict["scene_graphs"]["tot_obj_dense_splat"][:, -1]
        loss_occupancy = self.loss(reconstruction[:, -1], mask)
        loss_features = self.masked_loss(
            reconstruction[:, :-1],
            data_dict["scene_graphs"]["tot_obj_dense_splat"][:, :-1],
            mask.unsqueeze(1),
        )
        loss_dict = {
            "loss": loss_features + self.compute_decaying_weight() * loss_occupancy,
            "loss_features": loss_features,
            "loss_occupancy": loss_occupancy,
        }

        output_dict = {
            "reconstruction": reconstruction,
            "ground_truth": data_dict["scene_graphs"]["tot_obj_dense_splat"],
        }
        output_dict.update(data_dict)
        return output_dict, loss_dict

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            assert self.model is not None and isinstance(self.model, AutoEncoder)
            return self.train_step(epoch, iteration, data_dict)

    def set_eval_mode(self) -> None:
        self.training = False
        self.model.eval()
        self.loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self) -> None:
        self.training = True
        self.model.train()
        self.loss.train()
        torch.set_grad_enabled(True)

    def visualize(
        self, output_dict: Dict[str, Any], epoch: int, mode: str = "train"
    ) -> None:
        ground_truth_sparse = self._sparsify(output_dict["ground_truth"][:4])
        ground_truths = self.latent_autoencoder.decoder(ground_truth_sparse)

        reconstruction_sparse = self._sparsify(output_dict["reconstruction"][:4])
        if reconstruction_sparse is None:
            return
        reconstructions = self.latent_autoencoder.decoder(reconstruction_sparse)

        scene_ids = output_dict["scene_graphs"]["scene_ids"]
        obj_ids = output_dict["scene_graphs"]["obj_ids"]
        obj_count = output_dict["scene_graphs"]["graph_per_obj_count"]
        intrinsic = output_dict["scene_graphs"]["obj_intrinsics"]
        img_poses = output_dict["scene_graphs"]["obj_img_poses"]
        translations = output_dict["scene_graphs"]["mean_obj_splat"]
        scales = output_dict["scene_graphs"]["scale_obj_splat"]
        predicted_images = []
        ground_truth_images = []

        count = 0
        scene_idx = 0
        for i in range(len(reconstructions)):
            count += 1
            if count > obj_count[scene_idx]:
                scene_idx += 1
                count = 0
            scene_id = scene_ids[scene_idx][0]
            intrinsics = intrinsic[scene_id]
            extrinsics = img_poses[scene_id][obj_ids[i]][
                np.random.randint(0, len(img_poses[scene_id][obj_ids[i]]))
            ]

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
            reconstructions[i].rescale(
                torch.tensor([2, 2, 2], device=reconstructions[i].get_xyz.device)
            )
            reconstructions[i].translate(
                -torch.tensor([1, 1, 1], device=reconstructions[i].get_xyz.device)
            )
            reconstructions[i].rescale(scales[i])
            reconstructions[i].translate(translations[i])

            ground_truths[i].rescale(
                torch.tensor([2, 2, 2], device=reconstructions[i].get_xyz.device)
            )
            ground_truths[i].translate(
                -torch.tensor([1, 1, 1], device=reconstructions[i].get_xyz.device)
            )
            ground_truths[i].rescale(scales[i])
            ground_truths[i].translate(translations[i])

            rendered_image = render(
                viewpoint_camera,
                reconstructions[i],
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            ground_truth_image = render(
                viewpoint_camera,
                ground_truths[i],
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]
            predicted_images.append(rendered_image)
            ground_truth_images.append(ground_truth_image)
            reconstructions[i].save_ply(
                f"{self.cfg.output_dir}/events/{mode}_reconstruction_{i}.ply"
            )
            ground_truths[i].save_ply(f"{self.cfg.output_dir}/events/{mode}_gt_{i}.ply")

        predicted_images = torch.stack(predicted_images)
        ground_truth_images = torch.stack(ground_truth_images)

        save_image(
            predicted_images,
            f"{self.cfg.output_dir}/events/{mode}_predicted_images.png",
        )
        save_image(
            ground_truth_images,
            f"{self.cfg.output_dir}/events/{mode}_ground_truth_images.png",
        )

        side_by_side_images = torch.concat(
            [ground_truth_images, predicted_images],
            dim=-1,
        )
        self.writer.add_image(
            f"{mode}/reconstructions",
            side_by_side_images,
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
