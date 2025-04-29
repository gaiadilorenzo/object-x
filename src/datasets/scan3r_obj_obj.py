import json
import os
import os.path as osp
import random
import sys
import zipfile

import numpy as np
import torch
import torch.utils.data as data

from configs import Config
from src.modules.sparse.basic import SparseTensor, sparse_batch_cat, sparse_cat
from utils import common, scan3r

_THRESHOLD = 50
_MAX_PER_SCENE = 1000


def read_transform_matrix(root_dir):
    config_file = osp.join(root_dir, "files", "3RScan.json")
    rescan2ref = {}
    with open(config_file, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(
                        scans["transform"]
                    ).reshape(4, 4)
    return rescan2ref


class Scan3RObjObjDataset(data.Dataset):
    def __init__(self, cfg: Config, split: str):
        self.cfg = cfg

        self.seed = cfg.seed
        random.seed(self.seed)

        self.split = split
        self.use_predicted = cfg.autoencoder.encoder.use_predicted
        self.pc_resolution = cfg.val.pc_res if split == "val" else cfg.train.pc_res
        self.gs_resolution = 64
        self.anchor_type_name = cfg.data.preprocess.anchor_type_name
        self.scan_type = cfg.autoencoder.encoder.scan_type
        self.data_root_dir = cfg.data.root_dir

        scan_dirname = "out"
        scan_dirname = (
            osp.join(scan_dirname, "predicted") if self.use_predicted else scan_dirname
        )

        self.rescan2ref = read_transform_matrix(cfg.data.root_dir)

        self.scans_file_dir_orig = osp.join(cfg.data.root_dir, "files")
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, "scenes")
        self.scans_files_dir = osp.join(self.scans_dir, "files")

        self.mode = "orig" if self.split == "train" else cfg.val.data_mode

        self.anchor_data_filename = osp.join(
            self.scans_files_dir,
            "{}/anchors{}_{}.json".format(self.mode, self.anchor_type_name, split),
        )
        print(
            "[INFO] Reading from {} with point cloud resolution - {}".format(
                self.anchor_data_filename, self.pc_resolution
            )
        )
        self.anchor_data = common.load_json(self.anchor_data_filename)[:]

        if split == "val":
            final_anchor_data = []
            for anchor_data_idx in self.anchor_data:
                if (
                    cfg.val.overlap_low == cfg.val.overlap_high
                    or (
                        anchor_data_idx["overlap"] >= cfg.val.overlap_low
                        and anchor_data_idx["overlap"] < cfg.val.overlap_high
                    )
                    and os.path.exists(
                        os.path.join(
                            self.scans_file_dir_orig,
                            "gs_annotations",
                            anchor_data_idx["src"],
                        )
                    )
                    and os.path.exists(
                        os.path.join(
                            self.scans_file_dir_orig,
                            "gs_annotations",
                            anchor_data_idx["ref"],
                        )
                    )
                ):

                    final_anchor_data.append(anchor_data_idx)

            self.anchor_data = sorted(final_anchor_data, key=lambda x: x["src"])

        self.is_training = self.split == "train"
        self.do_augmentation = False if self.split == "val" else True

        self.rot_factor = cfg.train.rot_factor
        self.augment_noise = cfg.train.augmentation_noise

        # Jitter
        self.scale = 0.01
        self.clip = 0.05

        # Random Rigid Transformation
        self._rot_mag = 45.0
        self._trans_mag = 0.5

        # load 3D obj semantic annotations
        self.obj_3D_anno = {}

        self.objs_config_file = osp.join(self.scans_files_dir, "objects.json")
        self.gs_embedding_dir = osp.join(cfg.data.root_dir, "files", "gs_embeddings")
        objs_configs = common.load_json(self.objs_config_file)["scans"]

        scans_objs_info = {}
        for scan_item in objs_configs:
            scan_id = scan_item["scan"]
            objs_info = scan_item["objects"]
            scans_objs_info[scan_id] = objs_info

        for obj in objs_configs:
            scan_id = obj["scan"]
            self.obj_3D_anno[scan_id] = {}
            for obj_item in scans_objs_info[scan_id]:
                obj_id = int(obj_item["id"])
                obj_nyu_category = int(obj_item["nyu40"])
                # get object attributes and semantic category and concatenate them to form a sentence
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, obj_nyu_category)

    def __len__(self):
        return len(self.anchor_data)

    def _load_splats(self, scan_id: str, obj_id: int) -> dict:
        data_dict = {}
        gs_path = os.path.join(
            self.scans_file_dir_orig,
            "gs_annotations",
            scan_id,
            str(obj_id),
            f"voxel_output_dense.npz",
        )

        try:
            file = np.load(gs_path, mmap_mode="r")
            gs = torch.from_numpy(file["arr_0"]).float()
            coords = gs[:, :3]
            feats = gs[:, 3:]
            assert feats.shape[-1] == 1024
        except (FileNotFoundError, AssertionError, KeyError, zipfile.BadZipFile):
            print(f"File not found {gs_path}")
            coords = torch.zeros((1, 3), device="cpu")
            feats = torch.zeros((coords.shape[0], 1024), device="cpu")

        splat = SparseTensor(feats=feats, coords=coords.int())

        mean_scale_path = os.path.join(
            self.scans_files_dir,
            "gs_annotations",
            scan_id,
            str(obj_id),
            f"mean_scale_dense.npz",
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

    def __getitem__(self, idx):
        graph_data = self.anchor_data[idx]
        src_scan_id = graph_data["src"]
        ref_scan_id = graph_data["ref"]
        scan_id = src_scan_id.split("_")[0]

        overlap = graph_data["overlap"] if "overlap" in graph_data else -1.0

        # Centering
        src_points, _ = scan3r.load_plydata_npy(
            osp.join(self.scans_scenes_dir, "{}/data.npy".format(src_scan_id)),
            obj_ids=None,
        )
        ref_points, _ = scan3r.load_plydata_npy(
            osp.join(self.scans_scenes_dir, "{}/data.npy".format(ref_scan_id)),
            obj_ids=None,
        )
        if self.split == "train":
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(src_points, axis=0)
            else:
                pcl_center = np.mean(ref_points, axis=0)
        else:
            pcl_center = np.mean(src_points, axis=0)

        src_data_dict = common.load_pkl_data(
            osp.join(
                self.scans_files_dir, "{}/data/{}.pkl".format(self.mode, src_scan_id)
            )
        )
        ref_data_dict = common.load_pkl_data(
            osp.join(
                self.scans_files_dir, "{}/data/{}.pkl".format(self.mode, ref_scan_id)
            )
        )

        src_object_ids = src_data_dict["objects_id"][:_MAX_PER_SCENE]
        ref_object_ids = ref_data_dict["objects_id"][:_MAX_PER_SCENE]
        anchor_obj_ids = (
            graph_data["anchorIds"] if "anchorIds" in graph_data else src_object_ids
        )
        global_object_ids = np.concatenate(
            (src_data_dict["objects_cat"], ref_data_dict["objects_cat"])
        )

        anchor_obj_ids = [
            anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id != 0
        ]
        anchor_obj_ids = [
            anchor_obj_id
            for anchor_obj_id in anchor_obj_ids
            if anchor_obj_id in src_object_ids and anchor_obj_id in ref_object_ids
        ]

        if self.split == "train":
            anchor_cnt = (
                2
                if int(0.3 * len(anchor_obj_ids)) < 1
                else int(0.3 * len(anchor_obj_ids))
            )
            anchor_obj_ids = anchor_obj_ids[:anchor_cnt]

        src_edges = src_data_dict["edges"]
        ref_edges = ref_data_dict["edges"]

        src_object_points = (
            src_data_dict["obj_points"][self.pc_resolution] - pcl_center
        )[:_MAX_PER_SCENE]

        ref_object_points = (
            ref_data_dict["obj_points"][self.pc_resolution] - pcl_center
        )[:_MAX_PER_SCENE]

        edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

        src_object_id2idx = src_data_dict["object_id2idx"]

        e1i_idxs = np.array(
            [src_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]
        )  # e1i
        e1j_idxs = np.array(
            [
                src_object_id2idx[object_id]
                for object_id in src_object_ids
                if object_id not in anchor_obj_ids
            ]
        )  # e1j
        ref_object_id2idx = ref_data_dict["object_id2idx"]
        e2i_idxs = (
            np.array(
                [ref_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]
            )
            + src_object_points.shape[0]
        )  # e2i
        e2j_idxs = (
            np.array(
                [
                    ref_object_id2idx[object_id]
                    for object_id in ref_object_ids
                    if object_id not in anchor_obj_ids
                ]
            )
            + src_object_points.shape[0]
        )  # e2j

        tot_object_points = torch.cat(
            [torch.from_numpy(src_object_points), torch.from_numpy(ref_object_points)]
        ).type(torch.FloatTensor)
        tot_bow_vec_obj_edge_feats = torch.cat(
            [
                torch.from_numpy(src_data_dict["bow_vec_object_edge_feats"]),
                torch.from_numpy(ref_data_dict["bow_vec_object_edge_feats"]),
            ]
        )
        if not self.use_predicted:
            tot_bow_vec_obj_attr_feats = torch.cat(
                [
                    torch.from_numpy(src_data_dict["bow_vec_object_attr_feats"]),
                    torch.from_numpy(ref_data_dict["bow_vec_object_attr_feats"]),
                ]
            )

        else:
            tot_bow_vec_obj_attr_feats = torch.zeros(tot_object_points.shape[0], 41)

        tot_rel_pose = torch.cat(
            [
                torch.from_numpy(src_data_dict["rel_trans"]),
                torch.from_numpy(ref_data_dict["rel_trans"]),
            ]
        )
        data_dict = {}

        splat_dict = [
            self._load_splats(src_scan_id, obj_id) for obj_id in src_object_ids
        ] + [self._load_splats(ref_scan_id, obj_id) for obj_id in ref_object_ids]

        data_dict["tot_obj_splat"] = sparse_batch_cat(
            [s["tot_obj_splat"] for s in splat_dict]
        )
        data_dict["mean_obj_splat"] = torch.stack(
            [s["mean_obj_splat"] for s in splat_dict]
        )
        data_dict["scale_obj_splat"] = torch.stack(
            [s["scale_obj_splat"] for s in splat_dict]
        )
        data_dict["tot_rel_pose"] = tot_rel_pose.double()
        data_dict["obj_ids"] = np.concatenate([src_object_ids, ref_object_ids])
        data_dict["tot_obj_pts"] = tot_object_points
        assert data_dict["tot_obj_splat"].shape[0] == tot_object_points.shape[0]
        data_dict["graph_per_obj_count"] = np.array(
            [src_object_points.shape[0], ref_object_points.shape[0]]
        )
        data_dict["graph_per_edge_count"] = np.array(
            [src_edges.shape[0], ref_edges.shape[0]]
        )

        data_dict["e1i"] = e1i_idxs
        data_dict["e1i_count"] = e1i_idxs.shape[0]
        data_dict["e2i"] = e2i_idxs
        data_dict["e2i_count"] = e2i_idxs.shape[0]
        data_dict["e1j"] = e1j_idxs
        data_dict["e1j_count"] = e1j_idxs.shape[0]
        data_dict["e2j"] = e2j_idxs
        data_dict["e2j_count"] = e2j_idxs.shape[0]

        data_dict["tot_obj_count"] = tot_object_points.shape[0]
        data_dict["tot_bow_vec_object_attr_feats"] = tot_bow_vec_obj_attr_feats
        data_dict["tot_bow_vec_object_edge_feats"] = tot_bow_vec_obj_edge_feats
        data_dict["tot_rel_pose"] = tot_rel_pose
        data_dict["edges"] = edges

        data_dict["global_obj_ids"] = global_object_ids
        data_dict["scene_ids"] = [src_scan_id, ref_scan_id]
        data_dict["pcl_center"] = pcl_center
        data_dict["overlap"] = overlap

        return data_dict

    def _collate_entity_idxs(self, batch):
        e1i = np.concatenate([data["e1i"] for data in batch])
        e2i = np.concatenate([data["e2i"] for data in batch])
        e1j = np.concatenate([data["e1j"] for data in batch])
        e2j = np.concatenate([data["e2j"] for data in batch])

        e1i_start_idx = 0
        e2i_start_idx = 0
        e1j_start_idx = 0
        e2j_start_idx = 0
        prev_obj_cnt = 0

        for idx in range(len(batch)):
            e1i_end_idx = e1i_start_idx + batch[idx]["e1i_count"]
            e2i_end_idx = e2i_start_idx + batch[idx]["e2i_count"]
            e1j_end_idx = e1j_start_idx + batch[idx]["e1j_count"]
            e2j_end_idx = e2j_start_idx + batch[idx]["e2j_count"]

            e1i[e1i_start_idx:e1i_end_idx] += prev_obj_cnt
            e2i[e2i_start_idx:e2i_end_idx] += prev_obj_cnt
            e1j[e1j_start_idx:e1j_end_idx] += prev_obj_cnt
            e2j[e2j_start_idx:e2j_end_idx] += prev_obj_cnt

            e1i_start_idx, e2i_start_idx, e1j_start_idx, e2j_start_idx = (
                e1i_end_idx,
                e2i_end_idx,
                e1j_end_idx,
                e2j_end_idx,
            )
            prev_obj_cnt += batch[idx]["tot_obj_count"]

        e1i = e1i.astype(np.int32)
        e2i = e2i.astype(np.int32)
        e1j = e1j.astype(np.int32)
        e2j = e2j.astype(np.int32)

        return e1i, e2i, e1j, e2j

    def _collate_feats(self, batch, key, type_="torch_cat"):
        if type_ == "torch_cat":
            feats = torch.cat([data[key] for data in batch])
        elif type_ == "sparse_cat":
            feats = sparse_cat([data[key] for data in batch])
        elif type_ == "torch_stack":
            feats = torch.stack([data[key] for data in batch])
        return feats

    def collate_fn(self, batch):
        batch = [data for data in batch if data is not None]
        if len(batch) == 0:
            return None
        tot_object_points = self._collate_feats(batch, "tot_obj_pts")
        tot_bow_vec_object_attr_feats = self._collate_feats(
            batch, "tot_bow_vec_object_attr_feats"
        )
        tot_bow_vec_object_edge_feats = self._collate_feats(
            batch, "tot_bow_vec_object_edge_feats"
        )

        tot_rel_pose = self._collate_feats(batch, "tot_rel_pose")

        data_dict = {}
        data_dict
        data_dict["tot_obj_pts"] = tot_object_points
        data_dict["tot_rel_pose"] = tot_rel_pose.double()

        (
            data_dict["e1i"],
            data_dict["e2i"],
            data_dict["e1j"],
            data_dict["e2j"],
        ) = self._collate_entity_idxs(batch)

        data_dict["e1i_count"] = np.stack([data["e1i_count"] for data in batch])
        data_dict["e2i_count"] = np.stack([data["e2i_count"] for data in batch])
        data_dict["e1j_count"] = np.stack([data["e1j_count"] for data in batch])
        data_dict["e2j_count"] = np.stack([data["e2j_count"] for data in batch])
        data_dict["tot_obj_count"] = np.stack([data["tot_obj_count"] for data in batch])
        data_dict["global_obj_ids"] = np.concatenate(
            [data["global_obj_ids"] for data in batch]
        )
        data_dict[
            "tot_bow_vec_object_attr_feats"
        ] = tot_bow_vec_object_attr_feats.double()
        data_dict[
            "tot_bow_vec_object_edge_feats"
        ] = tot_bow_vec_object_edge_feats.double()

        data_dict["tot_obj_splat"] = self._collate_feats(
            batch, "tot_obj_splat", "sparse_cat"
        )

        data_dict["graph_per_obj_count"] = np.stack(
            [data["graph_per_obj_count"] for data in batch]
        )
        data_dict["graph_per_edge_count"] = np.stack(
            [data["graph_per_edge_count"] for data in batch]
        )
        data_dict["edges"] = self._collate_feats(batch, "edges")
        data_dict["scene_ids"] = np.stack([data["scene_ids"] for data in batch])
        data_dict["obj_ids"] = np.concatenate([data["obj_ids"] for data in batch])
        data_dict["pcl_center"] = np.stack([data["pcl_center"] for data in batch])

        data_dict["overlap"] = np.stack([data["overlap"] for data in batch])
        data_dict["batch_size"] = data_dict["overlap"].shape[0]

        scene_graph = {}
        scene_graph["scene_graphs"] = data_dict

        return scene_graph
