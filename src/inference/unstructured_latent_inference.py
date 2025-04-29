import argparse
import copy
import logging
import os
import os.path as osp

import imageio
import numpy as np
import open3d as o3d
import torch
from gaussian_renderer import render
from PIL import Image
from scene.cameras import MiniCam
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets import Scan3RSceneGraphDataset
from src.models.autoencoder import AutoEncoder
from src.models.latent_autoencoder import LatentAutoencoder
from src.modules.sparse.basic import SparseTensor
from src.representations import Gaussian
from utils import common, mesh, torch_util
from utils.gaussian_splatting import GaussianSplat
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov

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


class SceneGraph2UnstructuredLatentPipeline:
    def __init__(
        self,
        cfg: Config,
        visualize=False,
        split="train",
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.dataset = Scan3RSceneGraphDataset(cfg, split)
        self.vis = visualize
        self.rep_config = _REPRESENTATION_CONFIG
        self.output_dir = osp.join(cfg.data.root_dir, cfg.inference.output_dir)

    def load_model(self):
        self.latent_autoencoder = LatentAutoencoder(
            cfg=self.cfg.autoencoder, device=self.device
        )
        self.model = AutoEncoder(cfg=self.cfg.autoencoder, device=self.device)
        self.latent_autoencoder.load_state_dict(
            torch.load(
                self.cfg.inference.slat_model_path,
                map_location="cpu" if not torch.cuda.is_available() else "cuda",
            )["model"],
            strict=False,
        )
        self.model.load_state_dict(
            torch.load(
                self.cfg.inference.ulat_model_path,
                map_location="cpu" if not torch.cuda.is_available() else "cuda",
            )["model"],
            strict=False,
        )
        self.rep_config = self.latent_autoencoder.decoder.rep_config
        return self.model

    def _densify(self, sparse_splat, fill=0.0):
        shape = sparse_splat.dense().shape
        dense_splat = torch.full(
            (shape[0], 1, shape[2], shape[3], shape[4]),
            fill,
            device=sparse_splat.device,
        )
        dense_splat[
            sparse_splat.coords[:, 0],
            :,
            sparse_splat.coords[:, 1],
            sparse_splat.coords[:, 2],
            sparse_splat.coords[:, 3],
        ] = 1.0
        return torch.cat((sparse_splat.dense(), dense_splat), dim=1)

    def _sparsify(self, dense_splat):
        coords = torch.nonzero(dense_splat[:, -1] > 0.1, as_tuple=False)
        feats = dense_splat[coords[:, 0], :-1, coords[:, 1], coords[:, 2], coords[:, 3]]
        return SparseTensor(
            coords=coords.int(),
            feats=feats,
        )

    def inference(self, idx, object_id=None):
        with torch.no_grad():
            _LOGGER.info(f"Getting item: {idx}")
            data_dict = self.dataset.collate_fn([self.dataset[idx]])
            data_dict = (
                torch_util.to_cuda(data_dict)
                if torch.cuda.is_available()
                else data_dict
            )
            # Stage 1.
            sparse_splat = self.latent_autoencoder.encode(data_dict)

            # Stage 2.
            data_dict["scene_graphs"]["tot_obj_dense_splat"] = self._densify(
                sparse_splat
            )
            embedding = self.model.encode(data_dict)
            # Stage 2.
            reconstruction_dense = self.model.decode(embedding)
            reconstruction_sparse = self._sparsify(reconstruction_dense)

            # Stage 1.
            reconstruction = self.latent_autoencoder.decode(reconstruction_sparse)

            means, scales, obj_ids, scan_id = (
                data_dict["scene_graphs"]["mean_obj_splat"],
                data_dict["scene_graphs"]["scale_obj_splat"],
                data_dict["scene_graphs"]["obj_ids"],
                data_dict["scene_graphs"]["scene_ids"][0],
            )

        self.save_embedding(embedding, means, scales, obj_ids, scan_id)

        if self.vis:
            if object_id is not None:
                object_id = torch.tensor(object_id)
                idx = torch.where(
                    torch.isin(
                        torch.from_numpy(data_dict["scene_graphs"]["obj_ids"]),
                        object_id,
                    )
                )[0]
                if len(idx) == 0:
                    _LOGGER.warning(f"Object {object_id} not found in the scene.")
                    return
            else:
                _LOGGER.info(f"Saving all objects in the scene.")
                idx = torch.arange(len(reconstruction))
            self.save_scene(
                [copy.deepcopy(reconstruction[i]) for i in idx],
                scan_id,
                means[idx],
                scales[idx],
            )
            self.save_render_orbit(
                [copy.deepcopy(reconstruction[i]) for i in idx],
                scan_id,
                means[idx],
                scales[idx],
            )
            self.save_render_orbit_gs(scan_id, obj_ids[idx], means[idx], scales[idx])

    def save_embedding(self, embedding, means, scales, obj_ids, scan_id):
        output_path = osp.join(self.output_dir, f"{scan_id[0]}_ulat.npz")
        _LOGGER.info(f"Saving to {output_path}")
        np.savez(
            output_path,
            embedding=embedding.cpu(),
            mean=means.cpu(),
            scale=scales.cpu(),
            obj_id=obj_ids,
        )

    def save_scene(self, reconstruction, scan_id, means, scales):
        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                return None
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
            [splat._xyz for splat in reconstruction if splat is not None]
        )
        representation._features_dc = torch.concatenate(
            [splat._features_dc for splat in reconstruction if splat is not None]
        )
        representation._opacity = torch.concatenate(
            [splat._opacity for splat in reconstruction if splat is not None]
        )
        representation._scaling = torch.concatenate(
            [splat._scaling for splat in reconstruction if splat is not None]
        )
        representation._rotation = torch.concatenate(
            [splat._rotation for splat in reconstruction if splat is not None]
        )
        representation.save_ply(f"vis/{scan_id[0]}_joint.ply")

    def save_render(self, reconstruction, scene_ids, means, scales):
        if not osp.exists("vis/rendered"):
            os.makedirs("vis/rendered")

        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                return splat
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
            [reconstruction[i]._xyz for i in range(len(reconstruction))]
        )
        representation._features_dc = torch.concatenate(
            [reconstruction[i]._features_dc for i in range(len(reconstruction))]
        )
        representation._opacity = torch.concatenate(
            [reconstruction[i]._opacity for i in range(len(reconstruction))]
        )
        representation._scaling = torch.concatenate(
            [reconstruction[i]._scaling for i in range(len(reconstruction))]
        )
        representation._rotation = torch.concatenate(
            [reconstruction[i]._rotation for i in range(len(reconstruction))]
        )
        predicted_images = []
        ground_truth_images = []
        rendered_frames = []
        scene_id = scene_ids[0]
        intrinsics = self.dataset.image_intrinsics[scene_id]
        poses = self.dataset.image_poses[scene_id]

        gs_mesh = mesh.splat_to_mesh(
            splat=copy.deepcopy(representation).to_pt(),
            Ks=intrinsics["intrinsic_mat"],
            world_to_cams=poses,
            width=intrinsics["width"],
            height=intrinsics["height"],
            sh_degree_to_use=0,
            near_plane=0.01,
            far_plane=100.0,
        )

        for i, frame_id in enumerate(poses):
            extrinsics = poses[frame_id]
            image = Image.open(
                f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{frame_id}.color.jpg"
            )
            image = (
                torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            )  # SHape: (3, H, W)

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

            rendered_image = render(
                viewpoint_camera,
                representation,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            predicted_images.append(rendered_image)
            ground_truth_images.append(image)

            frame_path = f"vis/rendered/{scene_id}_frame_{i:03d}.png"
            save_image(rendered_image, frame_path)
            rendered_frames.append(
                (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
            )

            del image
            torch.cuda.empty_cache()

        video_path = f"vis/rendered/{scene_id}_rendered.mp4"
        imageio.mimsave(video_path, rendered_frames, fps=30)

    def save_render_orbit(
        self, reconstruction, scene_ids, means, scales, output_dir="vis/rendered"
    ):
        os.makedirs(output_dir, exist_ok=True)
        scene_id = scene_ids[0]

        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                print(f"Splat {i} is empty.")
                return splat
            splat.rescale(
                torch.tensor([2, 2, 2], device=reconstruction[i].get_xyz.device)
            )
            splat.translate(
                -torch.tensor([1, 1, 1], device=reconstruction[i].get_xyz.device)
            )
            splat.rescale(scales[i])
            splat.translate(means[i])
            return splat

        # Apply transformation to all objects
        reconstruction = list(map(lambda x: _scale(x), enumerate(reconstruction)))

        # Create the representation
        representation = Gaussian(
            sh_degree=0,
            aabb=[-0.0, -0.0, -0.0, 1.0, 1.0, 1.0],
            mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
            scaling_bias=self.rep_config["scaling_bias"],
            opacity_bias=self.rep_config["opacity_bias"],
            scaling_activation=self.rep_config["scaling_activation"],
        )

        representation._xyz = torch.cat(
            [reconstruction[i]._xyz for i in range(len(reconstruction))]
        )
        representation._features_dc = torch.cat(
            [reconstruction[i]._features_dc for i in range(len(reconstruction))]
        )
        representation._opacity = torch.cat(
            [reconstruction[i]._opacity for i in range(len(reconstruction))]
        )
        representation._scaling = torch.cat(
            [reconstruction[i]._scaling for i in range(len(reconstruction))]
        )
        representation._rotation = torch.cat(
            [reconstruction[i]._rotation for i in range(len(reconstruction))]
        )

        # Compute the centroid of all objects (Center of the scene)
        object_positions = representation._xyz.detach().cpu().numpy()
        scene_center = object_positions.mean(axis=0)  # Mean position of objects

        # Set camera path parameters
        num_frames = 120  # Number of frames for a smooth orbit
        radius = 2.0  # Distance of the camera from the scene center
        height = 1.0  # Keep the camera at the same height as the scene center
        angle_step = 2 * np.pi / num_frames  # Step size for rotation

        rendered_frames = []

        # Get intrinsics for frustum visualization
        intrinsics = self.dataset.image_intrinsics[scene_id]
        poses = np.stack(list(self.dataset.image_poses[scene_id].values()))

        gs_mesh = mesh.splat_to_mesh(
            splat=copy.deepcopy(representation).to_pt(),
            Ks=intrinsics["intrinsic_mat"],
            world_to_cams=poses,
            width=intrinsics["width"],
            height=intrinsics["height"],
            sh_degree_to_use=0,
            near_plane=0.01,
            far_plane=100.0,
        )
        o3d.io.write_triangle_mesh(f"{output_dir}/{scene_id}_mesh.ply", gs_mesh)

        for i in range(num_frames):
            theta = i * angle_step
            # Camera position: Move in a circular orbit around the Y-axis
            cam_x = scene_center[0] + radius * np.cos(theta)  # Orbit on XZ plane
            cam_y = scene_center[2] + radius * np.sin(theta)  # Orbit on XZ plane
            cam_z = scene_center[1] + height  # Keep at a fixed height

            # Camera should always look at the scene center
            cam_position = np.array([cam_x, cam_y, cam_z])  # Camera position
            forward = cam_position - scene_center  # Look at scene center
            forward /= np.linalg.norm(forward)  # Normalize forward vector

            # Define world UP direction
            up = np.array([0, 1, 0])  # Fixed Y-up

            # Compute RIGHT vector (perpendicular to FORWARD and UP)
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)  # Normalize right vector

            # Recompute UP to maintain perfect orthogonality
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)  # Normalize up vector

            # Construct the camera rotation matrix
            R = np.stack([right, up, -forward], axis=1)  # Rotation matrix
            T = cam_position  # Camera position

            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T
            viewmat = np.linalg.inv(viewmat)

            # Create camera instance
            viewpoint_camera = MiniCam(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
                znear=0.01,
                zfar=100.0,
                R=viewmat[:3, :3].T,
                T=viewmat[:3, 3],
            )

            rendered_image = render(
                viewpoint_camera,
                representation,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            frame_path = f"{output_dir}/{scene_ids[0]}_frame_{i:03d}.png"
            save_image(rendered_image, frame_path)
            rendered_frames.append(
                (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
            )

            torch.cuda.empty_cache()

        # Save video
        video_path = f"{output_dir}/{scene_ids[0]}_orbit_rendered.mp4"
        imageio.mimsave(video_path, rendered_frames, fps=30)
        print(f"Video saved at {video_path}")

    def save_render_orbit_gs(
        self, scene_ids, obj_ids, means, scales, output_dir="vis/rendered_gs"
    ):
        os.makedirs(output_dir, exist_ok=True)
        scene_id = scene_ids[0]
        obj_ids = obj_ids if type(obj_ids) == list else [obj_ids]

        # load gaussian splat representation
        reconstruction = [
            GaussianSplat.load_ply(
                f"{self.cfg.data.root_dir}/files/gs_annotations/{scene_id}/{obj_id}/point_cloud/iteration_7000/point_cloud.ply"
            ).to_torch()
            for obj_id in obj_ids
        ]

        # Create the representation
        representation = GaussianSplat(
            xyz=torch.cat([splat.xyz for splat in reconstruction]),
            features_dc=torch.cat([splat.features_dc for splat in reconstruction]),
            opacity=torch.cat([splat.opacity for splat in reconstruction]),
            scaling=torch.cat([splat.scaling for splat in reconstruction]),
            rotation=torch.cat([splat.rotation for splat in reconstruction]),
        )

        # Compute the centroid of all objects (Center of the scene)
        object_positions = representation.xyz.detach().cpu().numpy()
        scene_center = object_positions.mean(axis=0)  # Mean position of objects

        print(f"Scene Center: {scene_center}")

        # Set camera path parameters
        num_frames = 120  # Number of frames for a smooth orbit
        radius = 2.0  # Distance of the camera from the scene center
        height = 1.5  # Keep the camera at the same height as the scene center
        angle_step = 2 * np.pi / num_frames  # Step size for rotation

        rendered_frames = []

        # Get intrinsics for frustum visualization
        intrinsics = self.dataset.image_intrinsics[scene_id]
        poses = np.stack(list(self.dataset.image_poses[scene_id].values()))

        gs_mesh = mesh.splat_to_mesh(
            splat=copy.deepcopy(representation).to_pt(),
            Ks=intrinsics["intrinsic_mat"],
            world_to_cams=poses,
            width=intrinsics["width"],
            height=intrinsics["height"],
            sh_degree_to_use=0,
            near_plane=0.01,
            far_plane=100.0,
        )
        o3d.io.write_triangle_mesh(f"{output_dir}/{scene_id}_mesh.ply", gs_mesh)

        for i in range(num_frames):
            theta = i * angle_step
            # Camera position: Move in a circular orbit around the Y-axis
            cam_x = scene_center[0] + radius * np.cos(theta)  # Orbit on XZ plane
            cam_y = scene_center[2] + radius * np.sin(theta)  # Orbit on XZ plane
            cam_z = scene_center[1] + height  # Keep at a fixed height

            # Camera should always look at the scene center
            cam_position = np.array([cam_x, cam_y, cam_z])  # Camera position
            forward = cam_position - scene_center  # Look at scene center
            forward /= np.linalg.norm(forward)  # Normalize forward vector

            # Define world UP direction
            up = np.array([0, 1, 0])  # Fixed Y-up

            # Compute RIGHT vector (perpendicular to FORWARD and UP)
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)  # Normalize right vector

            # Recompute UP to maintain perfect orthogonality
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)  # Normalize up vector

            # Construct the camera rotation matrix
            R = np.stack([right, up, -forward], axis=1)  # Rotation matrix
            T = cam_position  # Camera position

            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T
            viewmat = np.linalg.inv(viewmat)

            # Create camera instance
            viewpoint_camera = MiniCam(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
                znear=0.01,
                zfar=100.0,
                R=viewmat[:3, :3].T,
                T=viewmat[:3, 3],
            )

            rendered_image = render(
                viewpoint_camera,
                representation,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            frame_path = f"{output_dir}/{scene_ids[0]}_frame_{i:03d}.png"
            save_image(rendered_image, frame_path)
            rendered_frames.append(
                (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
            )

            torch.cuda.empty_cache()

        # Save video
        video_path = f"{output_dir}/{scene_ids[0]}_orbit_rendered.mp4"
        imageio.mimsave(video_path, rendered_frames, fps=30)
        print(f"Video saved at {video_path}")

    def run(self, scene_id=None, object_id=None):
        if scene_id is not None:
            self.inference(scene_id, object_id)
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
    parser.add_argument(
        "--split", type=str, default="train", help="Split to use for inference."
    )
    parser.add_argument("--scene_id", type=str, default=None, help="Specific scene.")
    parser.add_argument(
        "--objects_id", type=int, nargs="+", default=None, help="Specific object."
    )
    parser.add_argument("--split", type=str, default="train", help="Specific split.")
    return parser.parse_known_args()


def main() -> None:
    """Run training."""

    common.init_log(level=logging.INFO)
    args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args, do_ensure_dir=False)
    pipeline = SceneGraph2UnstructuredLatentPipeline(
        cfg, visualize=args.visualize, split=args.split
    )
    pipeline.run(args.scene_id, args.objects_id)


if __name__ == "__main__":
    main()
