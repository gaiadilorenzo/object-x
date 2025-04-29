import argparse
import logging
import os.path as osp

import numpy as np
import torch

from configs import Config, update_configs
from src.datasets import Scan3RSceneGraphDataset
from src.models.latent_autoencoder import LatentAutoencoder
from src.representations import Gaussian
from utils import common, torch_util

_REPRESENTATION_CONFIG = {
    "perturb_offset": True,
    "voxel_size": 1.5,
    "num_gaussians": 32,
    "2d_filter_kernel_size": 0.1,
    "3d_filter_kernel_size": 9e-4,
    "scaling_bias": 4e-3,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
}

_LOGGER = logging.getLogger(__name__)


class SceneGraph2StructuredLatentPipeline:
    def __init__(self, cfg: Config, visualize=False, split="train"):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.dataset = Scan3RSceneGraphDataset(cfg, split)
        self.vis = visualize
        self.rep_config = _REPRESENTATION_CONFIG
        self.output_dir = osp.join(cfg.data.root_dir, cfg.inference.output_dir)

    def load_model(self):
        model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)
        model.load_state_dict(torch.load(self.cfg.inference.slat_model_path)["model"])
        self.rep_config = model.decoder.rep_config
        return model

    def inference(self, idx):
        with torch.no_grad():
            data_dict = self.dataset.collate_fn([self.dataset[idx]])
            data_dict = (
                torch_util.to_cuda(data_dict)
                if torch.cuda.is_available()
                else data_dict
            )
            if data_dict["scene_graphs"]["tot_obj_splat"].shape[0] > 100:
                _LOGGER.info(
                    f"Skipping {data_dict['scene_graphs']['scene_ids'][0]} due to large number of objects"
                )
                return
            embedding = self.model.encode(data_dict)
            means, scales, obj_ids, scan_id = (
                data_dict["scene_graphs"]["mean_obj_splat"],
                data_dict["scene_graphs"]["scale_obj_splat"],
                data_dict["scene_graphs"]["obj_ids"],
                data_dict["scene_graphs"]["scene_ids"][0],
            )
            reconstruction = self.model.decode(embedding)

        self.save_embedding(embedding, means, scales, obj_ids, scan_id)

        if self.vis:
            self.save_scene(reconstruction, scan_id, means, scales)

    def save_embedding(self, embedding, means, scales, obj_ids, scan_id):
        output_path = osp.join(self.output_dir, f"{scan_id[0]}_slat.npz")
        _LOGGER.info(f"Saving to {output_path}")
        np.savez(
            output_path,
            coords=embedding.coords.cpu(),
            feats=embedding.feats.cpu(),
            mean=means.cpu(),
            scale=scales.cpu(),
            obj_id=obj_ids,
        )

    def save_scene(self, reconstruction, scan_id, means, scales):
        def _scale(x):
            i, splat = x
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 1 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(
                torch.tensor([2, 2, 2], device=reconstruction[i].get_xyz.device)
            )
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 2.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.translate(
                -torch.tensor([1, 1, 1], device=reconstruction[i].get_xyz.device)
            )
            assert (
                splat._xyz.min() >= -1.0 - 1e-2 and splat._xyz.max() <= 1.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(scales[i])
            splat.translate(means[i])
            return splat

        reconstruction = list(map(lambda x: _scale(x), enumerate(reconstruction)))
        representation = Gaussian(
            sh_degree=0,
            aabb=[-0.0, -0.0, -0.0, 1.0, 1.0, 1.0],
            mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
            scaling_bias=self.rep_config["scaling_bias"],
            opacity_bias=self.rep_config["opacity_bias"],
            scaling_activation=self.rep_config["scaling_activation"],
        )
        representation._xyz = torch.concatenate(
            [splat._xyz for splat in reconstruction]
        )
        representation._features_dc = torch.concatenate(
            [splat._features_dc for splat in reconstruction]
        )
        representation._opacity = torch.concatenate(
            [splat._opacity for splat in reconstruction]
        )
        representation._scaling = torch.concatenate(
            [splat._scaling for splat in reconstruction]
        )
        representation._rotation = torch.concatenate(
            [splat._rotation for splat in reconstruction]
        )
        _LOGGER.info(f"Saving vis to vis/{scan_id[0]}_slat.ply")
        representation.save_ply(f"vis/{scan_id[0]}_slat.ply")

    def run(self, scene_id=None):
        if scene_id is not None:
            self.inference(scene_id)
        else:
            for idx in range(len(self.dataset)):
                self.inference(idx)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train structured latent inference model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results."
    )

    parser.add_argument("--scene_id", type=str, default=None, help="Specific scene.")
    parser.add_argument("--split", type=str, default="train", help="Specific split.")
    return parser.parse_known_args()


def main() -> None:
    """Run training."""

    common.init_log(level=logging.INFO)
    args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args, do_ensure_dir=False)
    pipeline = SceneGraph2StructuredLatentPipeline(
        cfg, visualize=args.visualize, split=args.split
    )
    pipeline.run(args.scene_id)


if __name__ == "__main__":
    main()
