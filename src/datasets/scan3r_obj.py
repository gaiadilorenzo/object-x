import logging
import os
import os.path as osp
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, Union

import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data
import tqdm

from configs import Config
from src.modules.sparse.basic import SparseTensor, sparse_batch_cat
from utils import common, scan3r

_LOGGER = logging.getLogger(__name__)


def _load_frame_poses(
    scans_scenes_dir: str, scan_id: str, image_paths: Dict[int, str], scan_name: str
) -> Dict[int, np.ndarray]:
    frame_idxs = [frame_idx for frame_idx in image_paths]
    return {
        scan_name: scan3r.load_frame_poses(
            scans_scenes_dir, scan_id, tuple(frame_idxs), type="quat_trans"
        )
    }


class Scan3RObjectDataset(data.Dataset):
    def __init__(self, cfg: Config, split: str):
        self.cfg = cfg

        self.seed = cfg.seed
        random.seed(self.seed)

        self.split = split
        self.preload_masks = cfg.data.preload_masks
        self.suffix = "_dense" if cfg.data.from_gt else ""

        self.data_root_dir = cfg.data.root_dir
        self.scans_dir = cfg.data.root_dir
        self.scans_files_dir = osp.join(self.scans_dir, "files")
        self.mode = "orig" if self.split == "train" else cfg.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)

        self.scans_scenes_dir = osp.join(self.scans_dir, "scenes")
        self.rescan = cfg.data.rescan
        self.resplit = "resplit_" if cfg.data.resplit else ""

        self._load_scan_ids()
        self._load_images()
        self._load_extrinsics()
        self._load_depth_intrinsics()
        self._load_intrinsics()
        self._load_patches()
        self._load_scene_graphs()
        self._load_gt_annos()

        self.img_patch_feat_dim = self.cfg.autoencoder.encoder.img_patch_feat_dim
        self.obj_patch_num = self.cfg.data.scene_graph.obj_patch_num
        self.obj_topk = self.cfg.data.scene_graph.obj_topk
        self.use_pos_enc = self.cfg.autoencoder.encoder.use_pos_enc
        if self.split == "test":
            self.load_split_frames()
        self.data_items = self._generate_data_items()
        _LOGGER.info(f"Total data items: {len(self.data_items)}")

    def _load_patches(self):
        self.obj_img_patches_scan_tops = {}
        obj_img_patch_name = self.cfg.data.scene_graph.obj_img_patch
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Patches"):
            obj_visual_file = osp.join(
                self.scans_files_dir, obj_img_patch_name, scan_id + ".pkl"
            )
            self.obj_img_patches_scan_tops[scan_id] = common.load_pkl_data(
                obj_visual_file
            )

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
        split = self.split
        scan_info_file = osp.join(self.scans_files_dir, "3RScan.json")
        all_scan_data = common.load_json(scan_info_file)

        self.refscans2scans = {}
        self.scans2refscans = {}
        self.all_scans_split = []
        for scan_data in all_scan_data:
            ref_scan_id = scan_data["reference"]
            self.refscans2scans[ref_scan_id] = [ref_scan_id]
            self.scans2refscans[ref_scan_id] = ref_scan_id
            for scan in scan_data["scans"]:
                self.refscans2scans[ref_scan_id].append(scan["reference"])
                self.scans2refscans[scan["reference"]] = ref_scan_id

        ref_scans_split = np.genfromtxt(
            osp.join(
                self.scans_files_dir_mode, "{}_{}scans.txt".format(split, self.resplit)
            ),
            dtype=str,
        )
        self.all_scans_split = []
        for ref_scan in ref_scans_split:
            self.all_scans_split += self.refscans2scans[ref_scan]
        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split

        if self.cfg.mode == "debug_few_scan":
            self.scan_ids = self.scan_ids[: int(0.1 * len(self.scan_ids))]

    def _load_images(self):
        self.image_paths = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Images"):
            self.image_paths[scan_id] = scan3r.load_frame_paths(
                self.scans_dir, scan_id, self.cfg.data.img.img_step
            )

    def _load_extrinsics(self):
        with Pool(processes=cpu_count()) as p:
            self.image_poses = [
                value
                for value in p.starmap(
                    _load_frame_poses,
                    tqdm.tqdm(
                        [
                            (
                                self.cfg.data.root_dir,
                                scan_id,
                                self.image_paths[scan_id],
                                scan_id,
                            )
                            for scan_id in self.scan_ids
                        ],
                        desc="Extrinsics",
                    ),
                )
            ]
        self.image_poses = {k: v for d in self.image_poses for k, v in d.items()}

    def _load_intrinsics(self):
        self.image_intrinsics = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Intrinsics"):
            self.image_intrinsics[scan_id] = scan3r.load_intrinsics(
                self.scans_scenes_dir, scan_id
            )

    def _load_depth_intrinsics(self):
        self.image_depth_intrinsics = {}
        for scan_id in tqdm.tqdm(self.scan_ids, desc="Depth Intrinsics"):
            self.image_depth_intrinsics[scan_id] = scan3r.load_intrinsics(
                self.scans_scenes_dir, scan_id, type="depth"
            )

    def load_split_frames(self):
        test_train_test_splits = common.load_json(
            osp.join(self.scans_files_dir, "test_train_test_splits.json")
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

    def _load_scene_graphs(self):
        if self.split == "train":
            self.pc_resolution = self.cfg.train.pc_res
        else:
            self.pc_resolution = self.cfg.val.pc_res

        rel_dim = self.cfg.autoencoder.encoder.rel_dim
        if rel_dim == 41:
            sg_filename = "data"
        elif rel_dim == 9:
            sg_filename = "data_rel9"
        else:
            raise ValueError("Invalid rel_dim")

        self.obj_3D_anno = {}
        self.objs_config_file = osp.join(self.scans_files_dir, "objects.json")
        objs_configs = common.load_json(self.objs_config_file)["scans"]
        scans_objs_info = {}
        for scan_item in objs_configs:
            scan_id = scan_item["scan"]
            objs_info = scan_item["objects"]
            scans_objs_info[scan_id] = objs_info

        self.obj_nyu40_id2name = common.idx2name(
            osp.join(self.scans_files_dir, "scannet40_classes.txt")
        )

        for scan_id in tqdm.tqdm(self.scan_ids, desc="Scene Graphs"):
            self.obj_3D_anno[scan_id] = {}
            for obj_item in scans_objs_info[scan_id]:
                obj_id = int(obj_item["id"])
                obj_nyu_category = int(obj_item["nyu40"])
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, obj_nyu_category)

        self.scene_graphs = {}

        for scan_id in tqdm.tqdm(self.scan_ids):
            data_dict = self._process_scan(rel_dim, sg_filename, scan_id)
            self.scene_graphs[scan_id] = data_dict

    def _load_splats(self, scan_id: str, obj_id: int) -> dict:
        data_dict = {}
        gs_path = os.path.join(
            self.scans_files_dir,
            "gs_annotations",
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
            "gs_annotations",
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

    def _process_scan(self, rel_dim: int, sg_filename: str, scan_id: str) -> dict:
        points, _ = scan3r.load_plydata_npy(
            osp.join(self.scans_scenes_dir, "{}/data.npy".format(scan_id)),
            data_dir=self.data_root_dir,
            scan_id=scan_id,
            obj_ids=None,
        )
        pcl_center = np.mean(points, axis=0)
        scene_graph_dict = common.load_pkl_data(
            osp.join(
                self.scans_files_dir_mode,
                "{}/{}.pkl".format(sg_filename, scan_id),
            )
        )
        object_ids = scene_graph_dict["objects_id"]
        global_object_ids = scene_graph_dict["objects_cat"]
        edges = scene_graph_dict["edges"]
        object_points = scene_graph_dict["obj_points"][self.pc_resolution] - pcl_center

        object_points = torch.from_numpy(object_points).type(torch.FloatTensor)
        edges = torch.from_numpy(edges)

        if not self.cfg.data.scene_graph.use_predicted:
            bow_vec_obj_attr_feats = torch.from_numpy(
                scene_graph_dict["bow_vec_object_attr_feats"]
            )
        else:
            bow_vec_obj_attr_feats = torch.zeros(object_points.shape[0], rel_dim)
        bow_vec_obj_edge_feats = torch.from_numpy(
            scene_graph_dict["bow_vec_object_edge_feats"]
        )

        rel_pose = torch.from_numpy(scene_graph_dict["rel_trans"])
        data_dict = {}
        data_dict["obj_ids"] = object_ids
        data_dict["tot_obj_pts"] = object_points
        data_dict["graph_per_obj_count"] = np.array([object_points.shape[0]])
        data_dict["graph_per_edge_count"] = np.array([edges.shape[0]])
        data_dict["tot_obj_count"] = object_points.shape[0]
        data_dict["tot_bow_vec_object_attr_feats"] = bow_vec_obj_attr_feats
        data_dict["tot_bow_vec_object_edge_feats"] = bow_vec_obj_edge_feats
        data_dict["tot_rel_pose"] = rel_pose
        data_dict["edges"] = edges
        data_dict["global_obj_ids"] = global_object_ids
        data_dict["scene_ids"] = [scan_id]
        data_dict["pcl_center"] = pcl_center
        data_dict["label"] = [
            self.obj_nyu40_id2name[self.obj_3D_anno[scan_id][obj_id][2]]
            for i, obj_id in enumerate(object_ids)
        ]
        data_dict["mean_obj_pts"] = data_dict["tot_obj_pts"].mean(dim=1, keepdim=True)
        data_dict["std_obj_pts"] = data_dict["tot_obj_pts"].std(dim=1, keepdim=True)
        data_dict["tot_obj_pts"] = (
            data_dict["tot_obj_pts"] - data_dict["mean_obj_pts"]
        ) / (data_dict["std_obj_pts"] + 1e-6)

        obj_ids_to_skip = []
        for obj_id in object_ids:
            gs_path = os.path.join(
                self.scans_files_dir,
                "gs_annotations",
                scan_id,
                str(obj_id),
                f"voxel_output{self.suffix}.npz",
            )
            if not os.path.exists(gs_path):
                obj_ids_to_skip.append(obj_id)

        obj_dict = {
            obj_id: {
                "obj_ids": data_dict["obj_ids"][obj_idx : obj_idx + 1],
                "scene_ids": data_dict["scene_ids"],
                "label": data_dict["label"][obj_idx : obj_idx + 1],
            }
            for obj_idx, obj_id in enumerate(object_ids)
            if obj_id not in obj_ids_to_skip
        }
        return obj_dict

    def load_mesh(self, scan_id: str, obj_id: int = -1) -> torch.Tensor:
        mesh = scan3r.load_ply_data(
            data_dir=self.scans_scenes_dir,
            scan_id=scan_id,
            label_file_name="labels.instances.annotated.v2.ply",
        )
        x, y, z = mesh["vertex"]["x"], mesh["vertex"]["y"], mesh["vertex"]["z"]
        label = mesh["vertex"]["objectId"]
        points = np.stack([x, y, z], axis=1)
        points = torch.from_numpy(points).type(torch.FloatTensor)
        return points[label == obj_id] if obj_id != -1 else points

    def _generate_data_items(self) -> list:
        data_items = []
        for scan_id in tqdm.tqdm(self.scan_ids):
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
        scene_graphs_["label"] = self._aggregate(scene_graph_infos, "label", "np_stack")

        obj_img_patches = defaultdict(dict)
        obj_img_poses = defaultdict(dict)
        obj_intrinsics = {}
        obj_annos = defaultdict(dict)
        obj_img_top_frames = defaultdict(dict)
        obj_depth_intrinsics = {}

        for scan_obj in zip(scene_graphs_["scene_ids"], scene_graphs_["obj_ids"]):
            scan_id, obj_id = scan_obj
            scan_id = scan_id[0]
            obj_img_patches_scan_tops = self.obj_img_patches_scan_tops[scan_id]
            obj_img_patches_scan = obj_img_patches_scan_tops["obj_visual_emb"]
            obj_top_frames = obj_img_patches_scan_tops["obj_image_votes_topK"]
            if obj_id not in obj_top_frames:
                obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                obj_img_patches[scan_id][obj_id] = torch.from_numpy(
                    obj_img_patch_embs
                ).float()
                if self.use_pos_enc:
                    identity_pos = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(1, -1)
                    obj_img_poses[scan_id][obj_id] = torch.from_numpy(
                        identity_pos
                    ).float()
                continue

            obj_img_patch_embs_list = []
            obj_img_poses_list = []
            obj_frames = (
                obj_top_frames[obj_id][: self.obj_topk]
                if len(obj_top_frames[obj_id]) >= self.obj_topk
                else obj_top_frames[obj_id]
            )
            for frame_idx in obj_frames:
                if obj_img_patches_scan[obj_id][frame_idx] is not None:
                    embs_frame = obj_img_patches_scan[obj_id][frame_idx]
                    embs_frame = (
                        embs_frame.reshape(1, -1)
                        if embs_frame.ndim == 1
                        else embs_frame
                    )
                    obj_img_patch_embs_list.append(embs_frame)
                    if self.use_pos_enc:
                        obj_img_poses_list.append(self.image_poses[scan_id][frame_idx])

            if len(obj_img_patch_embs_list) == 0:
                obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                if self.use_pos_enc:
                    obj_img_poses_arr = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(1, -1)
            else:
                obj_img_patch_embs = np.concatenate(obj_img_patch_embs_list, axis=0)
                if self.use_pos_enc:
                    obj_img_poses_arr = np.stack(obj_img_poses_list, axis=0)

            obj_img_patches[scan_id][obj_id] = torch.from_numpy(
                obj_img_patch_embs
            ).float()

            obj_intrinsics[scan_id] = self.image_intrinsics[scan_id]
            obj_depth_intrinsics[scan_id] = self.image_depth_intrinsics[scan_id]
            obj_img_top_frames[scan_id] = obj_top_frames

            if self.preload_masks:
                scan_annos = common.load_pkl_data(self.obj_2D_annos_pathes[scan_id])
                for frames in obj_top_frames.values():
                    for frame in frames:
                        obj_annos[scan_id][frame] = scan_annos[frame]
                scene_graphs_["obj_annos"] = obj_annos

            if self.use_pos_enc:
                obj_img_poses[scan_id][obj_id] = torch.from_numpy(
                    obj_img_poses_arr
                ).float()

            scene_graphs_["obj_img_patches"] = obj_img_patches
            scene_graphs_["obj_img_top_frames"] = obj_img_top_frames
            scene_graphs_["obj_intrinsics"] = obj_intrinsics
            scene_graphs_["obj_depth_intrinsics"] = obj_depth_intrinsics
            if self.use_pos_enc:
                scene_graphs_["obj_img_poses"] = obj_img_poses

        if self.split == "test":
            obj_train_frames = defaultdict(dict)
            obj_test_frames = defaultdict(dict)
            obj_train_poses = defaultdict(dict)
            obj_test_poses = defaultdict(dict)
            obj_intrinsics = {}
            obj_depth_intrinsics = {}
            for scan_obj in zip(scene_graphs_["scene_ids"], scene_graphs_["obj_ids"]):
                scan_id, obj_id = scan_obj
                scan_id = scan_id[0]
                train_frames = self.train_frames[scan_id][obj_id]
                test_frames = self.test_frames[scan_id][obj_id]
                if len(train_frames) == 0 or len(test_frames) == 0:
                    print(f"{scan_id} and obj_id: {obj_id} has no train or test frames")
                    return None

                train_poses = [
                    self.image_poses[scan_id][frame] for frame in train_frames
                ]
                test_poses = [self.image_poses[scan_id][frame] for frame in test_frames]
                train_poses = torch.from_numpy(np.stack(train_poses)).float()
                test_poses = torch.from_numpy(np.stack(test_poses)).float()

                obj_test_frames[scan_id][obj_id] = test_frames
                obj_train_frames[scan_id][obj_id] = train_frames
                obj_train_poses[scan_id][obj_id] = train_poses
                obj_test_poses[scan_id][obj_id] = test_poses
                obj_intrinsics[scan_id] = self.image_intrinsics[scan_id]
                obj_depth_intrinsics[scan_id] = self.image_depth_intrinsics[scan_id]

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
            scene_graphs_["obj_depth_intrinsics"] = obj_depth_intrinsics

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
