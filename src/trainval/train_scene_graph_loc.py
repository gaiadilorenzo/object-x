import argparse
import logging
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from gaussian_renderer import render
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
from src.models.losses.room_retrieval import get_loss, get_val_room_retr_loss
from src.models.patch_sg_aligner import PatchSGIEAligner
from src.modules.sparse.basic import SparseTensor
from utils import common
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov
from utils.path_obj_pair_visualizer import PatchObjectPairVisualizer

_FILL = 0.0


class Trainer(EpochBasedTrainer):
    def __init__(
        self, cfg: Config, parser: Optional[argparse.ArgumentParser] = None
    ) -> None:
        super().__init__(cfg, parser)

        # cfg
        self.cfg = cfg

        # get device
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available.")
        self.device = torch.device("cuda")

        # generate model
        model = self.create_model(cfg)
        self.register_model(model)

        self.freeze(cfg)

        # optimizer
        optimizer = self.create_optim(cfg)
        self.register_optimizer(optimizer)

        # get data loader
        start_time = time.time()
        train_loader, val_loader = get_train_val_data_loader(
            cfg, dataset=Scan3RPatchObjectDataset
        )
        loading_time = time.time() - start_time
        message = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        self.recon_loss = nn.MSELoss()
        self.masked_loss = mse_mask_loss
        self.threshold = 0.5

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
        else:
            raise NotImplementedError(
                "Scheduler {} not implemented.".format(cfg.train.optim.scheduler)
            )
        self.register_scheduler(scheduler)
        self.loss = get_loss(cfg)
        self.val_loss = get_val_room_retr_loss(cfg)
        self.create_visualizer(cfg)
        self.logger.info("Initialisation Complete")

    def create_model(self, cfg: Config):
        if not self.cfg.data.img_encoding.use_feature:
            from mmdet.models import build_backbone

            backbone_cfg_file = cfg.autoencoder.encoder.backbone.cfg_file
            # ugly hack to load pretrained model, maybe there is a better way
            backbone_cfg = Config.fromfile(backbone_cfg_file)
            backbone_pretrained_file = cfg.autoencoder.encoder.backbone.pretrained
            backbone_cfg.autoencoder.encoder["backbone"][
                "pretrained"
            ] = backbone_pretrained_file
            backbone = build_backbone(backbone_cfg.autoencoder.encoder["backbone"])
        else:
            backbone = None

        # get patch object aligner
        drop = cfg.autoencoder.encoder.other.drop
        ## 2Dbackbone
        num_reduce = cfg.autoencoder.encoder.backbone.num_reduce
        backbone_dim = cfg.autoencoder.encoder.backbone.backbone_dim
        img_rotate = cfg.data.img_encoding.img_rotate
        ## scene graph encoder
        sg_modules = cfg.autoencoder.encoder.modules
        sg_rel_dim = cfg.autoencoder.encoder.rel_dim
        attr_dim = cfg.autoencoder.encoder.attr_dim
        img_patch_feat_dim = cfg.autoencoder.encoder.img_patch_feat_dim
        multi_view_aggregator = getattr(
            cfg.autoencoder.encoder, "multi_view_aggregator", None
        )
        use_pos_enc = getattr(cfg.autoencoder.encoder, "use_pos_enc", False)

        ## temporal
        self.use_temporal = cfg.train.loss.use_temporal
        ## global descriptor
        self.use_global_descriptor = cfg.train.loss.use_global_descriptor
        self.global_descriptor_dim = cfg.autoencoder.encoder.global_descriptor_dim
        self.autoencoder = AutoEncoder(cfg=self.cfg.autoencoder, device=self.device)
        self.model = PatchSGIEAligner(
            backbone,
            num_reduce,
            backbone_dim,
            img_rotate,
            cfg.autoencoder.encoder.patch.hidden_dims,
            cfg.autoencoder.encoder.patch.encoder_dim,
            cfg.autoencoder.encoder.patch.gcn_layers,
            cfg.autoencoder.encoder.obj.embedding_dim,
            cfg.autoencoder.encoder.obj.embedding_hidden_dims,
            cfg.autoencoder.encoder.obj.encoder_dim,
            sg_modules,
            sg_rel_dim,
            attr_dim,
            img_patch_feat_dim,
            drop,
            use_temporal=cfg.train.loss.use_temporal,
            use_global_descriptor=self.use_global_descriptor,
            global_descriptor_dim=self.global_descriptor_dim,
            multi_view_aggregator=multi_view_aggregator,
            img_emb_dim=cfg.autoencoder.encoder.img_emb_dim,
            obj_img_pos_enc=use_pos_enc,
            in_feature_dim=cfg.autoencoder.encoder.voxel.in_feature_dim,
            out_feature_dim=cfg.autoencoder.encoder.voxel.out_feature_dim,
            decoder_net=cfg.autoencoder.decoder.net,
            voxel_channels=cfg.autoencoder.encoder.voxel.channels,
            encoder=self.autoencoder.encoder if self.autoencoder is not None else None,
        )

        # load pretrained sgaligner if required
        if cfg.autoencoder.encoder.use_pretrained:
            assert os.path.isfile(
                cfg.autoencoder.encoder.pretrained
            ), f"Pretrained sgaligner not found at {cfg.autoencoder.encoder.pretrained}."
            sgaligner_dict = torch.load(
                cfg.autoencoder.encoder.pretrained, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(sgaligner_dict["model"], strict=False)
            snapshot_keys = set(sgaligner_dict["model"])
            model_keys = set(self.model.state_dict().keys())
            missing_keys = model_keys - snapshot_keys
            unexpected_keys = snapshot_keys - model_keys
            self.logger.warning(f"Missing keys: {missing_keys}")
            self.logger.warning(f"Unexpected keys: {unexpected_keys}")
            self.logger.info("Model loaded.")
            if self.autoencoder is not None:
                self.autoencoder.load_state_dict(sgaligner_dict["model"], strict=False)
                snapshot_keys = set(sgaligner_dict["model"])
                model_keys = set(self.autoencoder.state_dict().keys())
                missing_keys = model_keys - snapshot_keys
                unexpected_keys = snapshot_keys - model_keys
                self.logger.warning(f"Missing keys: {missing_keys}")
                self.logger.warning(f"Unexpected keys: {unexpected_keys}")
                self.logger.info("Model loaded.")

        # load snapshot if required
        if cfg.use_resume and os.path.isfile(cfg.resume):
            self.load_snapshot(cfg.resume)
        # model to cuda
        self.model.to(self.device)

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

        message = "Model description:\n" + str(self.model)
        self.logger.info(message)
        return self.model

    def create_optim(self, cfg: Config):
        # only optimise params that require grad
        params_register = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        autoencoder_params = list(
            filter(lambda p: p.requires_grad, self.autoencoder.parameters())
        )
        params_register = list(set(params_register) - set(autoencoder_params))
        params_register += autoencoder_params
        params = [{"params": params_register}]
        optimizer = optim.AdamW(
            params,
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay,
        )
        self.logger.info(
            f"Number of parameters: {sum(p.numel() for p in params_register)}"
        )
        self.logger.info(
            f"Optimized parameters (Patch Aligner): {[name for name, param in self.model.named_parameters() if param.requires_grad]}"
        )
        self.logger.info(
            f"Optimizer parameters (Autoencoder): {[name for name, param in self.autoencoder.named_parameters() if param.requires_grad]}"
        )
        return optimizer

    def create_visualizer(self, cfg):
        self.result_visualizer = (
            None if not cfg.train.use_vis else PatchObjectPairVisualizer(cfg)
        )

    def save_snapshot(self, filename):
        super().save_snapshot(filename)
        # save autoencoder
        model_state_dict = self.autoencoder.state_dict()
        # Remove '.module' prefix in DistributedDataParallel mode.
        if self.distributed:
            model_state_dict = OrderedDict(
                [(key[7:], value) for key, value in model_state_dict.items()]
            )

        # save model
        filename = osp.join(self.snapshot_dir, f"autoencoder_{filename}.pth.tar")
        state_dict = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model_state_dict,
        }
        torch.save(state_dict, filename)
        self.logger.info('Model saved to "{}"'.format(filename))

        # save snapshot
        snapshot_filename = osp.join(self.snapshot_dir, "autoencoder_snapshot.pth.tar")
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

    def freeze(self, cfg):
        # freeze backbone params if required
        self.free_backbone_epoch = cfg.train.optim.free_backbone_epoch
        if (
            self.free_backbone_epoch > 0
        ) and not self.cfg.data.img_encoding.use_feature:
            # if (self.free_backbone_epoch > 0):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        self.free_sgaligner_epoch = cfg.train.optim.free_sgaligner_epoch

        if self.free_sgaligner_epoch > 0:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.autoencoder.voxel_decoder.parameters():
                param.requires_grad = False
            self.logger.info("Autoencoder frozen.")
        self.free_voxel_epoch = cfg.train.optim.free_voxel_epoch

        if self.free_voxel_epoch > 0:
            for name, param in self.model.encoder.voxel_embedding.named_parameters():
                if "out_layer" not in name:
                    param.requires_grad = False
            for name, param in self.autoencoder.voxel_decoder.named_parameters():
                if "input_layer" not in name:
                    param.requires_grad = False
            self.logger.info("Voxel frozen.")

    def defreeze_backbone(self):
        if not self.cfg.data.img_encoding.use_feature:
            for param in self.model.backbone.parameters():
                param.requires_grad = True

    def defreeze_sgaligner(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True

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

    def model_forward(self, data_dict):

        if self.cfg.data.img_encoding.use_feature:
            embeddings = self.model.forward_with_patch_features(data_dict)
        else:
            embeddings = self.model(data_dict)
            if self.cfg.data.img_encoding.record_feature:
                patch_raw_features = (
                    embeddings["patch_raw_features"].detach().cpu().numpy()
                )
                for batch_i in range(data_dict["batch_size"]):
                    file_path = data_dict["patch_features_paths"][batch_i]
                    file_parent_dir = os.path.dirname(file_path)
                    common.ensure_dir(file_parent_dir)
                    np.save(file_path, patch_raw_features[batch_i])

        joint_embedding = embeddings["obj_features"]
        graph_obj_count = data_dict["scene_graphs"]["graph_per_obj_count"]
        reconstruction = self.autoencoder.decode(
            joint_embedding[: graph_obj_count[0].item()]
        )
        output_dict = {
            "reconstruction": reconstruction,
            "ground_truth": data_dict["scene_graphs"]["tot_obj_dense_splat"][
                : graph_obj_count[0].item()
            ],
        }
        output_dict.update(embeddings)
        return output_dict

    def train_step(self, epoch, iteration, data_dict):
        sparse_splat = data_dict["scene_graphs"]["tot_obj_splat"]
        data_dict["scene_graphs"]["tot_obj_dense_splat"] = self._densify(sparse_splat)

        embeddings = self.model_forward(data_dict)

        loss_dict = self.loss(embeddings, data_dict)
        loss_emb = loss_dict.pop("loss")
        loss = loss_emb
        mask = embeddings["ground_truth"][:, -1]
        loss_occupancy = self.recon_loss(embeddings["reconstruction"][:, -1], mask)
        loss_features = self.masked_loss(
            embeddings["reconstruction"][:, :-1],
            embeddings["ground_truth"][:, :-1],
            mask.unsqueeze(1),
        )
        if self.free_sgaligner_epoch < 0:
            loss_recon = 100 * loss_features + loss_occupancy
            loss = loss_emb + loss_recon
        loss_dict = {
            "loss": loss,
            "loss_features": loss_features,
            "loss_occupancy": loss_occupancy,
            **loss_dict,
        }
        embeddings.update(data_dict)
        return embeddings, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        sparse_splat = data_dict["scene_graphs"]["tot_obj_splat"]
        data_dict["scene_graphs"]["tot_obj_dense_splat"] = self._densify(sparse_splat)

        embeddings = self.model_forward(data_dict)
        loss_dict = self.val_loss(embeddings, data_dict)
        loss_emb = loss_dict.pop("loss")
        loss = loss_emb
        mask = embeddings["ground_truth"][:, -1]
        loss_occupancy = self.recon_loss(embeddings["reconstruction"][:, -1], mask)
        loss_features = self.masked_loss(
            embeddings["reconstruction"][:, :-1],
            embeddings["ground_truth"][:, :-1],
            mask.unsqueeze(1),
        )
        if self.free_sgaligner_epoch < 0:
            loss_recon = 100 * loss_features + loss_occupancy
            loss = loss_emb + loss_recon
        loss_dict = {
            "loss": loss,
            "loss_features": loss_features,
            "loss_occupancy": loss_occupancy,
            **loss_dict,
        }
        embeddings.update(data_dict)
        return embeddings, loss_dict

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        self.val_loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        self.loss.train()
        torch.set_grad_enabled(True)

    def after_train_epoch(self, epoch):
        if epoch > self.cfg.train.optim.free_backbone_epoch:
            self.defreeze_backbone()
        if epoch > self.cfg.train.optim.free_sgaligner_epoch:
            self.defreeze_sgaligner()

    def visualize(self, output_dict, epoch, mode):
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

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        if self.cfg.train.use_vis:
            self.result_visualizer.visualize(data_dict, output_dict, epoch)

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        if self.cfg.train.use_vis:
            self.result_visualizer.visualize(data_dict, output_dict, epoch)


def parse_args(
    parser: argparse.ArgumentParser = None,
):
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
