import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data
import tqdm

from configs import Config
from src.modules.sparse.basic import SparseTensor, sparse_batch_cat
from utils import common, scannet

_LOGGER = logging.getLogger(__name__)


class ScannetObjectDataset(data.Dataset):
    def __init__(self, cfg: Config, split: str = "test"):
        self.cfg = cfg

        self.seed = cfg.seed
        random.seed(self.seed)

        self.split = split
        self.preload_masks = cfg.data.preload_masks
        self.load_mesh = True if self.split == "test" else False
        self.suffix = "_dense" if cfg.data.from_gt else ""
        self.data_root_dir = cfg.data.root_dir
        self.scans_scenes_dir = osp.join(cfg.data.root_dir, "scenes")
        self.scans_files_dir = osp.join(cfg.data.root_dir, "files")
        self.rescan = cfg.data.rescan
        self.img_step = cfg.data.img.img_step

        self.img_patch_feat_dim = self.cfg.autoencoder.encoder.img_patch_feat_dim
        self.obj_patch_num = self.cfg.data.scene_graph.obj_patch_num
        self.obj_topk = self.cfg.data.scene_graph.obj_topk
        self.use_pos_enc = self.cfg.autoencoder.encoder.use_pos_enc

        self._load_scan_ids()
        self._load_images()
        self._load_extrinsics()
        self._load_intrinsics()
        self._load_scene_graphs()
        self._load_gt_annos()

        self.load_split_frames()
        self.data_items = self._generate_data_items()
        _LOGGER.info(f"Total data items: {len(self.data_items)}")

    def _load_gt_annos(self):
        self.gt_2D_anno_folder = osp.join(
            self.scans_files_dir, "gt_projection/obj_id_pkl"
        )
        self.obj_2D_annos_pathes = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="2D annos"):
            self.obj_2D_annos_pathes[scan_id] = osp.join(
                self.gt_2D_anno_folder, "{}.pkl".format(scan_id)
            )

    def _load_scan_ids(self):
        subscan_ids_generated = np.genfromtxt(
            osp.join(self.cfg.data.root_dir, "files", "scannet_test_split.txt"),
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

        self.all_scans_split = [
            scan_id for scans in self.rooms_info.values() for scan_id in scans
        ]
        self.scan_ids = [
            next(
                (
                    room_scan
                    for room_scan in room_scans
                    if osp.exists(
                        osp.join(
                            self.cfg.data.root_dir, "scene_graph_fusion", room_scans[0]
                        )
                    )
                ),
                None,
            )
            for room_scans in self.rooms_info.values()
        ]

        self.scan_ids = [scan_id for scan_id in self.scan_ids if scan_id is not None]
        self.scan2room = {
            scan_id: room_id
            for room_id, scans in self.rooms_info.items()
            for scan_id in scans
        }

        if self.rescan:
            self.scan_ids = self.all_scans_split

        if self.cfg.mode == "debug_few_scan":
            self.scan_ids = self.scan_ids[: int(0.1 * len(self.scan_ids))]

        # TODO(gaia): due to timing constraints
        self.scan_ids = self.scan_ids[:77]
        _LOGGER.info(f"Total scans: {len(self.scan_ids)}")

    def _load_images(self):
        self.frame_idxs = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Frames"):
            frame_idxs = scannet.load_frame_idxs(
                self.data_root_dir, scan_id, self.img_step
            )
            self.frame_idxs[scan_id] = frame_idxs

    def _load_extrinsics(self):
        self.image_poses = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Extrinsics"):
            image_poses = scannet.load_frame_poses(
                self.data_root_dir, scan_id, self.img_step, type="quat_trans"
            )
            self.image_poses[scan_id] = image_poses

    def _load_intrinsics(self):
        self.image_intrinsics = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Intrinsics"):
            intrinsics = scannet.load_frame_intrinsics(self.data_root_dir, scan_id)
            self.image_intrinsics[scan_id] = intrinsics

    def _load_scene_graphs(self):
        sg_folder_name = "scene_graph_fusion"
        self.scene_graphs = {}
        self.obj_3D_anno = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Scene graphs"):
            if not osp.exists(
                osp.join(self.cfg.data.root_dir, sg_folder_name, scan_id)
            ):
                print(f"Scene graph not found for {scan_id}")
                continue
            sg_folder_scan = osp.join(self.cfg.data.root_dir, sg_folder_name, scan_id)
            points, _ = scannet.load_plydata_npy(osp.join(sg_folder_scan, "data.npy"))
            pcl_center = np.mean(points, axis=0)
            scene_graph_dict = common.load_pkl_data(
                osp.join(sg_folder_scan, "{}.pkl".format(scan_id))
            )
            object_ids = scene_graph_dict["objects_id"]
            global_object_ids = scene_graph_dict["objects_cat"]
            object_points = (
                scene_graph_dict["obj_points"][self.cfg.train.pc_res] - pcl_center
            )
            object_points = torch.from_numpy(object_points).type(torch.FloatTensor)

            data_dict = {}
            edges = scene_graph_dict["edges"]
            edges = torch.from_numpy(edges)
            if "bow_vec_object_edge_feats" in scene_graph_dict:
                bow_vec_obj_edge_feats = torch.from_numpy(
                    scene_graph_dict["bow_vec_object_edge_feats"]
                )
            else:
                rel_dim = self.cfg.autoencode.encoder.rel_dim
                bow_vec_obj_edge_feats = torch.zeros(edges.shape[0], rel_dim)
            data_dict["graph_per_edge_count"] = np.array([edges.shape[0]])
            data_dict["tot_bow_vec_object_edge_feats"] = bow_vec_obj_edge_feats
            rel_pose = torch.from_numpy(scene_graph_dict["rel_trans"])
            data_dict["tot_rel_pose"] = rel_pose
            data_dict["edges"] = edges
            if "bow_vec_object_attr_feats" in scene_graph_dict:
                bow_vec_obj_attr_feats = torch.from_numpy(
                    scene_graph_dict["bow_vec_object_attr_feats"]
                )
            else:
                attri_dim = self.cfg.autoencoder.encoder.attr_dim
                bow_vec_obj_attr_feats = torch.zeros(object_points.shape[0], attri_dim)
            data_dict["tot_bow_vec_object_attr_feats"] = bow_vec_obj_attr_feats

            data_dict["obj_ids"] = object_ids
            data_dict["tot_obj_pts"] = object_points
            data_dict["graph_per_obj_count"] = np.array([object_points.shape[0]])
            data_dict["tot_obj_count"] = object_points.shape[0]
            data_dict["scene_ids"] = [scan_id]
            data_dict["pcl_center"] = pcl_center

            self.obj_3D_anno[scan_id] = {}
            for idx, obj_id in enumerate(object_ids):
                self.obj_3D_anno[scan_id][obj_id] = (
                    scan_id,
                    obj_id,
                    global_object_ids[idx],
                )

            obj_ids_to_skip = []
            for obj_id in object_ids:
                gs_path = os.path.join(
                    self.cfg.data.root_dir,
                    "files",
                    "gs_annotations_scannet",
                    scan_id,
                    str(obj_id),
                    f"voxel_output{self.suffix}.npz",
                )
                if not os.path.exists(gs_path):
                    obj_ids_to_skip.append(obj_id)
                    continue

            if self.load_mesh:
                mesh = scannet.load_ply_data(
                    data_dir=self.cfg.data.root_dir,
                    scan_id=scan_id,
                )
                x, y, z = mesh["vertex"]["x"], mesh["vertex"]["y"], mesh["vertex"]["z"]
                label = mesh["vertex"]["label"]
                points = np.stack([x, y, z], axis=1)
                points = torch.from_numpy(points).type(torch.FloatTensor)

            obj_dict = {
                obj_id: {
                    "obj_ids": data_dict["obj_ids"][obj_idx : obj_idx + 1],
                    "scene_ids": data_dict["scene_ids"],
                    "tot_obj_pts": points[label == obj_id],
                }
                for obj_idx, obj_id in enumerate(object_ids)
                if obj_id not in obj_ids_to_skip
            }
            self.scene_graphs[scan_id] = obj_dict

    def load_split_frames(self):
        test_train_test_splits = common.load_json(
            osp.join(self.scans_files_dir, "test_train_test_splits_scannet.json")
        )
        self.test_frames = defaultdict(dict)
        self.train_frames = defaultdict(dict)
        for scan_id in self.scan_ids:
            for obj_id in test_train_test_splits[scan_id]:
                self.test_frames[scan_id][int(obj_id)] = test_train_test_splits[
                    scan_id
                ][obj_id]["test"]
                self.train_frames[scan_id][int(obj_id)] = test_train_test_splits[
                    scan_id
                ][obj_id]["train"]

    def _load_splats(self, scan_id: str, obj_id: int) -> dict:
        data_dict = {}
        gs_path = os.path.join(
            self.scans_files_dir,
            "gs_annotations_scannet",
            scan_id,
            str(obj_id),
            f"voxel_output{self.suffix}.npz",
        )
        try:
            file = np.load(gs_path, mmap_mode="r")
            gs = torch.from_numpy(file["arr_0"]).float()
            coords = gs[:, :3]
            feats = gs[:, 3:]
            assert feats.shape[-1] == 1024
        except (FileNotFoundError, AssertionError, KeyError):
            coords = torch.zeros((1, 3), device="cpu")
            feats = torch.zeros((coords.shape[0], 1024), device="cpu")
        splat = SparseTensor(feats=feats, coords=coords.int())
        mean_scale_path = os.path.join(
            self.scans_files_dir,
            "gs_annotations_scannet",
            scan_id,
            str(obj_id),
            f"mean_scale{self.suffix}.npz",
        )
        if os.path.exists(mean_scale_path):
            mean_scale = np.load(mean_scale_path)
            mean = torch.from_numpy(mean_scale["mean"]).float()
            scale = torch.from_numpy(mean_scale["scale"]).float()
        else:
            mean = torch.zeros((3)).float()
            scale = torch.ones(()).float()

        data_dict["tot_obj_splat"] = splat
        data_dict["mean_obj_splat"] = mean
        data_dict["scale_obj_splat"] = scale
        return data_dict

    def _generate_data_items(self) -> list:
        data_items = []
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Data items"):
            if scan_id not in self.scene_graphs:
                continue
            for obj_id in self.scene_graphs[scan_id]:
                data_item_dict = {}
                data_item_dict["scan_id"] = scan_id
                data_item_dict["obj_id"] = obj_id
                data_items.append(data_item_dict)
        return data_items

    def _item_to_dict(self, data_item: dict) -> dict:
        data_dict = {}
        scan_id = data_item["scan_id"]
        data_dict["scan_id"] = scan_id
        data_dict["obj_id"] = data_item["obj_id"]
        return data_dict

    def _aggregate(
        self, data_dict: dict, key: str, mode: str
    ) -> Union[torch.Tensor, np.ndarray]:
        if mode == "torch_cat":
            return torch.cat([data[key] for data in data_dict])
        elif mode == "torch_stack":
            return torch.stack([data[key] for data in data_dict])
        elif mode == "np_concat":
            return np.concatenate([data[key] for data in data_dict])
        elif mode == "np_stack":
            return np.stack([data[key] for data in data_dict])
        elif mode == "sparse_cat":
            return sparse_batch_cat([data[key] for data in data_dict])
        else:
            raise NotImplementedError

    def _collate(self, batch: list) -> dict:
        scans_batch = [data["scan_id"] for data in batch if data is not None]
        objs_batch = [data["obj_id"] for data in batch if data is not None]
        batch_size = len(batch)
        data_dict = {}
        data_dict["batch_size"] = batch_size
        data_dict["scan_ids"] = np.stack(
            [data["scan_id"] for data in batch if data is not None]
        )
        scene_graph_infos = [
            self.scene_graphs[scan_id][obj_id]
            for scan_id, obj_id in zip(scans_batch, objs_batch)
        ]
        scene_graphs_ = {}
        scans_size = len(scene_graph_infos)
        scene_graphs_["batch_size"] = scans_size
        scene_graphs_["obj_ids"] = self._aggregate(
            scene_graph_infos, "obj_ids", "np_concat"
        )
        splat_dict = [
            self._load_splats(scan_id, obj_id)
            for scan_id, obj_id in zip(scans_batch, objs_batch)
        ]
        scene_graphs_["tot_obj_splat"] = self._aggregate(
            splat_dict, "tot_obj_splat", "sparse_cat"
        ).float()
        scene_graphs_["mean_obj_splat"] = self._aggregate(
            splat_dict, "mean_obj_splat", "torch_stack"
        ).float()
        scene_graphs_["scale_obj_splat"] = self._aggregate(
            splat_dict, "scale_obj_splat", "torch_stack"
        ).float()
        scene_graphs_["scene_ids"] = self._aggregate(
            scene_graph_infos, "scene_ids", "np_stack"
        )
        scene_graphs_["tot_obj_pts"] = {
            scan_id.item(): {
                obj_id: self.scene_graphs[scan_id.item()][obj_id]["tot_obj_pts"]
                for obj_id in self.scene_graphs[scan_id.item()]
            }
            for scan_id in scene_graphs_["scene_ids"]
        }
        obj_img_poses = defaultdict(dict)
        obj_intrinsics = {}
        obj_annos = defaultdict(dict)
        obj_img_top_frames = defaultdict(dict)

        for scan_obj in zip(scene_graphs_["scene_ids"], scene_graphs_["obj_ids"]):
            scan_id, obj_id = scan_obj
            scan_id = scan_id[0]
            scan_annos = common.load_pkl_data(self.obj_2D_annos_pathes[scan_id])
            obj_top_frames = sorted(
                [
                    (frame_idx, (scan_annos[frame_idx] == obj_id).sum())
                    for frame_idx in scan_annos
                ],
                key=lambda x: x[1],
                reverse=True,
            )[: self.obj_topk]
            obj_img_top_frames[scan_id][obj_id] = [frame for frame, _ in obj_top_frames]
            obj_img_poses[scan_id][obj_id] = np.array(
                [
                    self.image_poses[scan_id][frame_idx]
                    for frame_idx, _ in obj_top_frames
                ]
            )
            obj_intrinsics[scan_id] = self.image_intrinsics[scan_id]
            scene_graphs_["obj_img_top_frames"] = obj_img_top_frames
            scene_graphs_["obj_intrinsics"] = obj_intrinsics
            if self.use_pos_enc:
                scene_graphs_["obj_img_poses"] = obj_img_poses

        obj_train_frames = defaultdict(dict)
        obj_test_frames = defaultdict(dict)
        obj_train_poses = defaultdict(dict)
        obj_test_poses = defaultdict(dict)
        obj_intrinsics = {}
        for scan_obj in zip(scene_graphs_["scene_ids"], scene_graphs_["obj_ids"]):
            scan_id, obj_id = scan_obj
            scan_id = scan_id[0]
            train_frames = self.train_frames[scan_id][obj_id]
            test_frames = self.test_frames[scan_id][obj_id]
            if len(train_frames) == 0 or len(test_frames) == 0:
                print(f"{scan_id} and obj_id: {obj_id} has no train or test frames")
                return None

            train_poses = [self.image_poses[scan_id][frame] for frame in train_frames]
            test_poses = [self.image_poses[scan_id][frame] for frame in test_frames]
            train_poses = torch.from_numpy(np.stack(train_poses)).float()
            test_poses = torch.from_numpy(np.stack(test_poses)).float()

            obj_test_frames[scan_id][obj_id] = test_frames
            obj_train_frames[scan_id][obj_id] = train_frames
            obj_train_poses[scan_id][obj_id] = train_poses
            obj_test_poses[scan_id][obj_id] = test_poses
            obj_intrinsics[scan_id] = self.image_intrinsics[scan_id]

            if self.preload_masks:
                scan_annos = common.load_pkl_data(self.obj_2D_annos_pathes[scan_id])
                for frame in test_frames + train_frames:
                    obj_annos[scan_id][frame] = scan_annos[frame]
                scene_graphs_["obj_annos"] = obj_annos

        scene_graphs_["obj_train_frames"] = obj_train_frames
        scene_graphs_["obj_test_frames"] = obj_test_frames
        scene_graphs_["obj_train_poses"] = obj_train_poses
        scene_graphs_["obj_test_poses"] = obj_test_poses
        scene_graphs_["obj_intrinsics"] = obj_intrinsics
        data_dict["scene_graphs"] = scene_graphs_
        if len(batch) > 0:
            return data_dict
        else:
            return None

    def __getitem__(self, idx: int) -> dict:
        data_dict = self._item_to_dict(self.data_items[idx])
        return data_dict

    def collate_fn(self, batch: list) -> dict:
        return self._collate(batch)

    def __len__(self) -> int:
        return len(self.data_items)
