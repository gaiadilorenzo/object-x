import argparse
import logging
import os.path as osp
import shutil
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from plyfile import PlyData
from tqdm.auto import tqdm

from configs import Config, update_configs
from utils import common, scannet

device = o3c.Device("CUDA", 0) if torch.cuda.is_available() else o3c.Device("CPU:0")
point_dtype = o3c.float32
dist_th = 0.05
mesh_file_name = "_vh_clean_2.labels.ply"
sg_pred_ply = "inseg_filtered.ply"

_LOGGER = logging.getLogger(__name__)


class ScannetIMGProjector:
    def __init__(self, cfg: Config, split: str = "test", vis: bool = False):
        self.cfg = cfg
        self.vis = vis

        self.split = split
        subscan_ids_generated = np.genfromtxt(
            osp.join(cfg.data.root_dir, "files", "scannet_test_split.txt"),
            dtype=str,
        )
        self.rooms_info = {
            room_id: [
                scan_id
                for scan_id in subscan_ids_generated
                if scan_id.startswith(room_id)
            ]
            for room_id in np.unique(
                [scan_id.split("_")[0] for scan_id in subscan_ids_generated]
            )
        }

        self.scan_ids = [
            scan_id for scans in self.rooms_info.values() for scan_id in scans
        ]
        self.scan2room = {
            scan_id: room_id
            for room_id, scans in self.rooms_info.items()
            for scan_id in scans
        }

        self.img_step = cfg.data.img.img_step
        self.img_paths = {}
        self.img_poses = {}
        self.color_intrinsics = {}
        for scan_id in tqdm(self.scan_ids, desc="Loading images."):
            img_paths = scannet.load_frame_paths(
                cfg.data.root_dir, scan_id, self.img_step
            )
            self.img_paths[scan_id] = img_paths

            poses = scannet.load_frame_poses(cfg.data.root_dir, scan_id, self.img_step)
            self.img_poses[scan_id] = poses

            intrinsic = scannet.load_frame_intrinsics(
                cfg.data.root_dir, scan_id, "color"
            )
            self.color_intrinsics[scan_id] = intrinsic

        self.mesh_files = {}
        for scan_id in self.scan_ids:
            mesh_file = osp.join(
                cfg.data.root_dir, "scenes", scan_id, scan_id + mesh_file_name
            )
            self.mesh_files[scan_id] = mesh_file

        self.pred_ply_files = {}
        for scan_id in self.scan_ids:
            pred_ply_file = osp.join(
                cfg.data.root_dir, "scene_graph_fusion", scan_id, sg_pred_ply
            )
            self.pred_ply_files[scan_id] = pred_ply_file

        self.save_dir = osp.join(self.cfg.data.root_dir, "files", "gt_projection")
        self.save_pkl_dir = osp.join(self.save_dir, "obj_id_pkl")
        common.ensure_dir(self.save_dir)
        common.ensure_dir(self.save_pkl_dir)

    def run(self):
        """Project the object labels from the scene graph fusion to the images."""

        for scan_id in tqdm(self.scan_ids, desc="Scans"):
            self._project(scan_id)

    def _project(self, scan_id: str):
        gt_patch_anno = {}

        mesh = o3d.io.read_triangle_mesh(self.mesh_files[scan_id])
        mesh_vertices = np.asarray(mesh.vertices)

        sgfusion_pcl_file = self.pred_ply_files[scan_id]
        if osp.exists(sgfusion_pcl_file) is False:
            print(f"File not found {sgfusion_pcl_file}")
            return
        sgfusion_data = PlyData.read(sgfusion_pcl_file)["vertex"]
        sgfusion_points = np.stack(
            [sgfusion_data["x"], sgfusion_data["y"], sgfusion_data["z"]], axis=1
        )
        sgfusion_labels = np.asarray(sgfusion_data["label"])

        # transfer ply labels to mesh by open3d knn search
        sgfusion_points_tensor = o3c.Tensor(
            sgfusion_points, dtype=point_dtype, device=device
        )
        kdtree_sgfusion = o3c.nns.NearestNeighborSearch(sgfusion_points_tensor)
        kdtree_sgfusion.knn_index()
        mesh_vertices_tensor = o3c.Tensor(
            mesh_vertices, dtype=point_dtype, device=device
        )
        [idx, dist] = kdtree_sgfusion.knn_search(mesh_vertices_tensor, 1)
        dist_arr = (dist.cpu().numpy()).reshape(-1)
        idx_arr = (idx.cpu().numpy()).reshape(-1)
        valid_idx = dist_arr < dist_th**2
        mesh_obj_labels = np.zeros(mesh_vertices.shape[0], dtype=np.int32)
        mesh_obj_labels[valid_idx] = sgfusion_labels[idx_arr[valid_idx]]

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        mesh_triangles_arr = np.asarray(mesh.triangles)
        mesh_colors = np.asarray(mesh.vertex_colors) * 255.0
        mesh_colors = mesh_colors.round()

        intrinsics_mat = self.color_intrinsics[scan_id]
        width = intrinsics_mat["width"]
        height = intrinsics_mat["height"]
        intrinsics = intrinsics_mat["intrinsic_mat"]

        num_triangles = mesh_triangles_arr.shape[0]
        obj_id_imgs = {}
        color_imgs = {}
        for frame_idx in tqdm(self.img_paths[scan_id], desc="Poses"):
            if frame_idx not in self.img_poses[scan_id]:
                continue
            pose_W_C = self.img_poses[scan_id][frame_idx]
            pose_C_W = np.linalg.inv(pose_W_C)
            color_map, obj_id_map = self.segment_result(
                scene,
                intrinsics,
                pose_C_W,
                width,
                height,
                mesh_triangles_arr,
                num_triangles,
                mesh_colors,
                mesh_obj_labels,
            )
            obj_id_imgs[frame_idx] = obj_id_map
            color_imgs[frame_idx] = color_map
            if self.vis is True:
                img_path = self.img_paths[scan_id][frame_idx]  # jpg file
                shutil.copy(img_path, "tmp/gt_image.png")
                cv2.imwrite("tmp/obj_map.png", obj_id_map)
                cv2.imwrite("tmp/color_map.png", color_map)
        save_scan_pkl_dir = osp.join(self.save_pkl_dir, "{}.pkl".format(scan_id))
        common.write_pkl_data(obj_id_imgs, save_scan_pkl_dir)
        return gt_patch_anno

    def segment_result(
        self,
        scene: o3d.t.geometry.RaycastingScene,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        width: int,
        height: int,
        mesh_triangles: np.ndarray,
        num_triangles: int,
        colors: np.ndarray,
        obj_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Segment the mesh by projecting the mesh triangles to the image plane
        and assign the object id to the pixels that hit the mesh triangles.

        Args:
            scene: open3d.t.geometry.RaycastingScene
            intrinsics: np.ndarray, shape (3, 3)
            extrinsics: np.ndarray, shape (4, 4)
            width: int
            height: int
            mesh_triangles: np.ndarray, shape (num_triangles, 3)
            num_triangles: int
            colors: np.ndarray, shape (num_points, 3)
            obj_ids: np.ndarray, shape (num_points)

        Returns:
            color_map: np.ndarray, shape (height, width, 3)
            obj_id_map: np.ndarray, shape (height, width)
        """

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsics.astype(np.float64),
            extrinsic_matrix=extrinsics.astype(np.float64),
            width_px=width,
            height_px=height,
        )

        ans = scene.cast_rays(rays)
        hit_triangles_ids = ans["primitive_ids"].numpy()
        hit_triangles_ids_valid_masks = hit_triangles_ids < num_triangles
        hit_triangles_ids_valid = hit_triangles_ids[hit_triangles_ids_valid_masks]
        hit_triangles_valid = mesh_triangles[hit_triangles_ids_valid]
        hit_points_ids_valid = hit_triangles_valid[:, 0]

        color_map = np.zeros((height, width, 3), dtype=np.uint8)  # for visualization
        obj_id_map = np.zeros((height, width), dtype=np.int32)  #
        color_map[hit_triangles_ids_valid_masks] = colors[hit_points_ids_valid]
        obj_id_map[hit_triangles_ids_valid_masks] = obj_ids[hit_points_ids_valid]
        return color_map, obj_id_map


def parse_args():
    parser = argparse.ArgumentParser(description="Ground Truth Annotations for ScanNet")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default="preprocessing/gt_anno/gt_anno_scannet.yaml",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
    )
    return parser.parse_known_args()


def main():
    common.init_log()
    _LOGGER.info("***** Generate GT Annotations for ScanNet *****")
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    scannet_dino_generator = ScannetIMGProjector(cfg, split="test", vis=args.vis)
    scannet_dino_generator.run()


if __name__ == "__main__":
    main()
