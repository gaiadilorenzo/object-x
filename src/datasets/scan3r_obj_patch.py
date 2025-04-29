import logging
import os
import os.path as osp
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data
import tqdm

from configs import Config
from src.modules.sparse.basic import SparseTensor, sparse_batch_cat, sparse_cat
from utils import common, scan3r

from .augumentation import ElasticDistortion

_LOGGER = logging.getLogger(__name__)


def _load_frame_poses(
    scans_scenes_dir: str, scan_id: str, image_paths: dict[int, str], scan_name: str
) -> dict[int, np.ndarray]:
    frame_idxs = [frame_idx for frame_idx in image_paths]
    return {
        scan_name: scan3r.load_frame_poses(
            scans_scenes_dir, scan_id, tuple(frame_idxs), type="quat_trans"
        )
    }


class Scan3RPatchObjectDataset(data.Dataset):
    def __init__(self, cfg: Config, split: str):
        self.cfg = cfg

        self.undefined = 0
        self.root_dir = self.cfg.data.root_dir

        self.seed = cfg.seed
        random.seed(self.seed)

        self.split = split
        self.use_predicted = cfg.autoencoder.encoder.use_predicted
        self.sgaligner_modules = cfg.autoencoder.encoder.modules

        self.data_root_dir = cfg.data.root_dir
        scan_dirname = "" if cfg.autoencoder.encoder.scan_type == "scan" else "out"
        scan_dirname = (
            osp.join(scan_dirname, "predicted") if self.use_predicted else scan_dirname
        )
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, "files")
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, "orig")

        if self.cfg.data.img_encoding.img_rotate:
            self.image_w = self.cfg.data.img_encoding.resize_h
            self.image_h = self.cfg.data.img_encoding.resize_w
            self.image_patch_w = self.cfg.data.img_encoding.patch_h
            self.image_patch_h = self.cfg.data.img_encoding.patch_w
        else:
            self.image_w = self.cfg.data.img.w
            self.image_h = self.cfg.data.img.h
            self.image_patch_w = self.cfg.data.img_encoding.patch_w
            self.image_patch_h = self.cfg.data.img_encoding.patch_h

        self.patch_w_size_int = int(self.image_w / self.image_patch_w)
        self.patch_h_size_int = int(self.image_h / self.image_patch_h)

        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate

        self.patch_h, self.patch_w = self.image_patch_h, self.image_patch_w
        self.preload_masks = self.cfg.data.preload_masks

        self.step = self.cfg.data.img.img_step
        self.num_patch = self.patch_h * self.patch_w
        self.suffix = "_dense" if cfg.data.from_gt else ""

        self.scans_scenes_dir = osp.join(self.scans_dir, "scenes")

        self.use_cross_scene = cfg.data.cross_scene.use_cross_scene
        self.num_scenes = cfg.data.cross_scene.num_scenes
        self.num_negative_samples = cfg.data.cross_scene.num_negative_samples
        self.use_tf_idf = cfg.data.cross_scene.use_tf_idf

        self.img_patch_feat_dim = self.cfg.autoencoder.encoder.img_patch_feat_dim
        self.obj_patch_num = self.cfg.data.scene_graph.obj_patch_num
        self.obj_topk = self.cfg.data.scene_graph.obj_topk
        self.use_pos_enc = self.cfg.autoencoder.encoder.use_pos_enc

        # if split is val, then use all object from other scenes as negative samples
        # if room_retrieval, then use load additional data items
        self.rescan = cfg.data.rescan
        self.room_retrieval = False

        if split == "val" or split == "test":
            self.num_negative_samples = -1
            self.num_scenes = 9
            self.room_retrieval = True
            self.room_retrieval_epsilon_th = cfg.val.room_retrieval.epsilon_th
            self.rescan = True
            self.step = 1

        self.temporal = cfg.data.temporal
        self.resplit = "resplit_" if cfg.data.resplit else ""

        self._load_scan_ids()
        self._load_images()
        self._load_extrinsics()
        self._load_intrinsics()
        self._load_patch_features()
        self._load_patch_annos()
        self._load_patches()
        self._load_scene_graphs()
        self._load_gt_annos()

        self.use_aug = cfg.train.data_aug.use_aug
        self.use_aug_3D = cfg.train.data_aug.use_aug_3D
        self.img_rot = cfg.train.data_aug.img.rotation
        self.img_Hor_flip = cfg.train.data_aug.img.horizontal_flip
        self.img_Ver_flip = cfg.train.data_aug.img.vertical_flip
        self.img_jitter = cfg.train.data_aug.img.color
        self.trans_2D = A.Compose(
            transforms=[
                A.VerticalFlip(p=self.img_Ver_flip),
                A.HorizontalFlip(p=self.img_Hor_flip),
                A.Rotate(
                    limit=int(self.img_rot),
                    p=0.8,
                    interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
            ]
        )
        color_jitter = self.img_jitter
        self.brightness_2D = A.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter,
        )

        self.elastic_distortion = ElasticDistortion(
            apply_distorsion=cfg.train.data_aug.use_aug_3D,
            granularity=cfg.train.data_aug.pcs.granularity,
            magnitude=cfg.train.data_aug.pcs.magnitude,
        )

        if self.split == "val" or self.split == "test":
            self.candidate_scans = {}
            for scan_id in self.scan_ids:
                self.candidate_scans[scan_id] = self.sample_candidate_scenes(
                    scan_id,
                    self.num_scenes,
                )

        if self.split == "test":
            self.load_split_frames()

        self.data_items = self.generate_data_items()
        _LOGGER.info(f"Total data items: {len(self.data_items)}")

    def _load_patch_annos(self):
        self.patch_anno = {}
        self.patch_anno_folder = osp.join(
            self.scans_files_dir, "patch_anno/patch_anno_16_9"
        )
        for scan_id in tqdm.tqdm(self.all_scans_split, desc="Patch Annos"):
            self.patch_anno[scan_id] = common.load_pkl_data(
                osp.join(self.patch_anno_folder, "{}.pkl".format(scan_id))
            )
            if len(self.patch_anno[scan_id]) == 0:
                print("patch anno length not match for {}".format(scan_id))

    def _load_patches(self):
        self.obj_img_patches_scan_tops = {}
        obj_img_patch_name = self.cfg.data.scene_graph.obj_img_patch
        for scan_id in tqdm.tqdm(self.all_scans_split, desc="Patches"):
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
        for scan_id in tqdm.tqdm(self.scan_ids, desc="GT 2D Annos"):
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
        for scan_id in tqdm.tqdm(self.all_scans_split, desc="Images"):
            self.image_paths[scan_id] = scan3r.load_frame_paths(
                self.scans_dir, scan_id, self.step
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
                                self.root_dir,
                                scan_id,
                                self.image_paths[scan_id],
                                scan_id,
                            )
                            for scan_id in self.all_scans_split
                        ],
                        desc="Extrinsics",
                    ),
                )
            ]
        self.image_poses = {k: v for d in self.image_poses for k, v in d.items()}

    def _load_intrinsics(self):
        self.image_intrinsics = {}
        for scan_id in tqdm.tqdm(self.all_scans_split, desc="Intrinsics"):
            self.image_intrinsics[scan_id] = scan3r.load_intrinsics(
                self.scans_scenes_dir, scan_id
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

    def _load_patch_features(self):
        self.use_2D_feature = self.cfg.data.img_encoding.use_feature
        self.preload_2D_feature = self.cfg.data.img_encoding.preload_feature
        self.patch_feature_folder = osp.join(
            self.scans_files_dir, self.cfg.data.img_encoding.feature_dir
        )
        self.patch_features = {}
        self.patch_features_paths = {}
        if self.use_2D_feature:
            if self.preload_2D_feature:
                for scan_id in tqdm.tqdm(self.all_scans_split, desc="Patch Features"):
                    self.patch_features[scan_id] = scan3r.load_patch_feature_scans(
                        self.data_root_dir,
                        self.patch_feature_folder,
                        scan_id,
                        self.step,
                    )
            else:
                for scan_id in tqdm.tqdm(self.all_scans_split, desc="Patch Features"):
                    self.patch_features_paths[scan_id] = osp.join(
                        self.patch_feature_folder, "{}.pkl".format(scan_id)
                    )

    def _load_splats(
        self, scan_id: str, obj_id: int = None, preload_slat: bool = True
    ) -> dict:
        data_dict = {}
        if preload_slat:
            gs_path = os.path.join(
                self.scans_files_dir,
                "gs_embeddings",
                f"{scan_id}_slat.npz",
            )
            file = np.load(gs_path, mmap_mode="r")
            coords = file["coords"]
            feats = file["feats"]
            mean = torch.from_numpy(file["mean"])
            scale = torch.from_numpy(file["scale"])
            splat = SparseTensor(
                feats=torch.from_numpy(feats).float(),
                coords=torch.from_numpy(coords).int(),
            )
            obj_id = file["obj_id"]
        else:
            splats = []
            means = []
            scales = []

            for obj_id in self.scene_graphs[scan_id]["obj_ids"]:
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
                    _LOGGER.warning(f"File not found {gs_path}")
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
                splats.append(splat)
                means.append(mean)
                scales.append(scale)
            splat = sparse_batch_cat(splats)
            mean = torch.stack(means)
            scale = torch.stack(scales)

        data_dict["mean_obj_splat"] = mean
        data_dict["scale_obj_splat"] = scale
        data_dict["tot_obj_splat"] = splat
        data_dict["obj_id"] = obj_id
        data_dict["scan_id"] = scan_id

        return data_dict

    def _load_scene_graphs(self):
        self.pc_resolution = (
            self.cfg.train.pc_res if self.split == "train" else self.cfg.val.pc_res
        )
        rel_dim = self.cfg.autoencoder.encoder.rel_dim
        sg_filename = (
            "data"
            if rel_dim == 41
            else "data_rel9"
            if rel_dim == 9
            else ValueError("Invalid rel_dim")
        )

        self.scene_graphs = {}
        for scan_id in tqdm.tqdm(self.all_scans_split, desc="Scene Graphs"):
            points, _ = scan3r.load_plydata_npy(
                osp.join(self.scans_scenes_dir, "{}/data.npy".format(scan_id)),
                obj_ids=None,
            )
            pcl_center = np.mean(points, axis=0)
            scene_graph_dict = common.load_pkl_data(
                osp.join(
                    self.scans_files_dir_mode, "{}/{}.pkl".format(sg_filename, scan_id)
                )
            )
            object_ids = scene_graph_dict["objects_id"]
            global_object_ids = scene_graph_dict["objects_cat"]
            edges = scene_graph_dict["edges"]
            object_points = (
                scene_graph_dict["obj_points"][self.pc_resolution] - pcl_center
            )
            object_points = torch.from_numpy(object_points).type(torch.FloatTensor)
            edges = torch.from_numpy(edges)
            bow_vec_obj_attr_feats = (
                torch.from_numpy(scene_graph_dict["bow_vec_object_attr_feats"])
                if not self.use_predicted
                else torch.zeros(object_points.shape[0], rel_dim)
            )
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
            self.scene_graphs[scan_id] = data_dict

        self.obj_3D_anno = {}
        self.objs_config_file = osp.join(self.scans_files_dir, "objects.json")
        objs_configs = common.load_json(self.objs_config_file)["scans"]
        scans_objs_info = {}
        for scan_item in objs_configs:
            scan_id = scan_item["scan"]
            objs_info = scan_item["objects"]
            scans_objs_info[scan_id] = objs_info
        for scan_id in self.all_scans_split:
            self.obj_3D_anno[scan_id] = {}
            for obj_item in scans_objs_info[scan_id]:
                obj_id = int(obj_item["id"])
                obj_nyu_category = int(obj_item["nyu40"])
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, obj_nyu_category)

        self.obj_nyu40_id2name = common.idx2name(
            osp.join(self.scans_files_dir, "scannet40_classes.txt")
        )

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

    def sample_candidate_scenes(self, scan_id: str, num_scenes: int) -> list:
        candidate_scans = []
        scans_same_scene = self.refscans2scans[self.scans2refscans[scan_id]]
        for scan in self.all_scans_split:
            if scan not in scans_same_scene:
                candidate_scans.append(scan)
        sampled_scans = random.sample(candidate_scans, num_scenes)
        return sampled_scans

    def sample_candidate_scenes_for_scans(
        self, scan_ids: list, num_scenes: int
    ) -> dict:
        candidate_scans = {}
        ref_scans = [self.scans2refscans[scan_id] for scan_id in scan_ids]
        ref_scans = list(set(ref_scans))
        num_ref_scans = len(ref_scans)
        additional_candidate_sample_pool = [
            scan
            for scan in self.all_scans_split
            if self.scans2refscans[scan] not in ref_scans
        ]
        additional_candidates = random.sample(
            additional_candidate_sample_pool, num_scenes
        )
        for scan_id in scan_ids:
            candidate_scans[scan_id] = list(set(additional_candidates))
        candidate_scans_all = list(
            set([scan for scan_list in candidate_scans.values() for scan in scan_list])
        )
        union_scans = list(set(scan_ids + candidate_scans_all))
        return candidate_scans, union_scans

    def _sample_temporal(self, scan_id: str) -> str:
        candidate_scans = []
        ref_scan = self.scans2refscans[scan_id]
        for scan in self.refscans2scans[ref_scan]:
            if scan != scan_id:
                candidate_scans.append(scan)
        if len(candidate_scans) == 0:
            return None
        else:
            sampled_scan = random.sample(candidate_scans, 1)[0]
            return sampled_scan

    def generate_data_items(self) -> list:
        data_items = []
        for scan_id in self.scan_ids:
            image_paths = self.image_paths[scan_id]
            i = 0
            for frame_idx in image_paths:
                if self.split != "test" and i % 5 != 0:
                    i += 1
                    continue
                i += 1
                data_item_dict = {}
                if self.use_2D_feature:
                    if self.preload_2D_feature:
                        data_item_dict["patch_features"] = self.patch_features[scan_id][
                            frame_idx
                        ]
                    else:
                        data_item_dict[
                            "patch_features_path"
                        ] = self.patch_features_paths[scan_id]
                else:
                    data_item_dict["img_path"] = image_paths[frame_idx]
                data_item_dict["frame_idx"] = frame_idx
                data_item_dict["scan_id"] = scan_id
                data_items.append(data_item_dict)
                if self.cfg.task == "reconstruction":
                    break
        return data_items

    def _item_to_dict(self, data_item: dict, temporal: bool = False) -> dict:
        data_dict = {}
        scan_id = data_item["scan_id"]
        if temporal:
            data_dict["scan_id_temporal"] = self._sample_temporal(scan_id)

        frame_idx = data_item["frame_idx"]

        if self.use_2D_feature:
            if self.preload_2D_feature:
                patch_features = data_item["patch_features"]
            else:
                patch_features = common.load_pkl_data(data_item["patch_features_path"])[
                    frame_idx
                ]
            if patch_features.ndim == 2:
                patch_features = patch_features.reshape(
                    self.patch_h, self.patch_w, self.img_patch_feat_dim
                )
        else:
            img_path = data_item["img_path"]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # type: ignore
            img = cv2.resize(
                img,
                (self.image_resize_w, self.image_resize_h),  # type: ignore
                interpolation=cv2.INTER_LINEAR,
            )  # type: ignore
            if self.img_rotate:
                img = img.transpose(1, 0, 2)
                img = np.flip(img, 1)
            if self.use_aug and self.split == "train":
                augments_2D = self.trans_2D(image=img, mask=obj_2D_anno)
                img = augments_2D["image"]
                obj_2D_anno = augments_2D["mask"]
                img = self.brightness_2D(image=img)["image"]

        patch_anno_frame = self.patch_anno[scan_id][frame_idx]
        if self.cfg.data.img_encoding.img_rotate:
            patch_anno_frame = patch_anno_frame.transpose(1, 0)
            patch_anno_frame = np.flip(patch_anno_frame, 1)
        obj_2D_patch_anno_flatten = patch_anno_frame.reshape(-1)

        data_dict["scan_id"] = scan_id
        data_dict["frame_idx"] = frame_idx
        data_dict["obj_2D_patch_anno_flatten"] = obj_2D_patch_anno_flatten
        if self.use_2D_feature:
            data_dict["patch_features"] = patch_features
        else:
            data_dict["image"] = img
            if self.cfg.data.img_encoding.record_feature:
                data_dict["patch_features_path"] = self.patch_features_paths[scan_id][
                    frame_idx
                ]
        return data_dict

    def _get_obj_patch_assoc_dict(
        self, data_item: dict, candidate_scans: list, sg_obj_idxs: dict
    ) -> dict:
        scan_id = data_item["scan_id"]
        if candidate_scans is None:
            candidate_scans_cur = []
        else:
            candidate_scans_cur = candidate_scans[scan_id]
        gt_2D_anno_flat = data_item["obj_2D_patch_anno_flatten"]
        assoc_data_dict = self._get_obj_patch_assoc_scan(
            scan_id, candidate_scans_cur, gt_2D_anno_flat, sg_obj_idxs
        )

        if self.temporal:
            scan_id_temporal = data_item["scan_id_temporal"]
            assoc_data_dict_temporal = self._get_obj_patch_assoc_scan(
                scan_id_temporal, candidate_scans_cur, gt_2D_anno_flat, sg_obj_idxs
            )
            return assoc_data_dict, assoc_data_dict_temporal
        else:
            return assoc_data_dict, None

    def _get_obj_patch_assoc_scan(
        self,
        scan_id: str,
        candidate_scans: list,
        gt_2D_anno_flat: np.ndarray,
        sg_obj_idxs: dict,
    ) -> dict:
        """
        Get object-patch association for a scan.
        Note: eij_matrix, (num_patch, num_3D_obj), record 2D-3D patch-object pairs
        Note: e2j_matrix, (num_patch, num_3D_obj), record 2D-3D patch-object negative pairs
        """

        obj_3D_idx2info = {}
        obj_3D_id2idx_cur_scan = {}
        scans_sg_obj_idxs = []
        candata_scan_obj_idxs = {}

        all_candi_scans = [scan_id] + candidate_scans
        N_scenes = 1 + len(candidate_scans)
        n_scenes_per_sem = {}
        n_words_per_scene = {candi_scan_id: 0 for candi_scan_id in all_candi_scans}
        n_word_scene = {candi_scan_id: {} for candi_scan_id in all_candi_scans}
        reweight_matrix_scans = {
            candi_scan_id: None for candi_scan_id in all_candi_scans
        }
        cadidate_scans_semantic_ids = []

        objs_ids_cur_scan = self.scene_graphs[scan_id]["obj_ids"]
        idx = 0
        for obj_id in objs_ids_cur_scan:
            obj_3D_idx2info[idx] = self.obj_3D_anno[scan_id][obj_id]
            obj_3D_id2idx_cur_scan[obj_id] = idx
            scans_sg_obj_idxs.append(sg_obj_idxs[scan_id][obj_id])
            cadidate_scans_semantic_ids.append(self.obj_3D_anno[scan_id][obj_id][2])
            if scan_id not in candata_scan_obj_idxs:
                candata_scan_obj_idxs[scan_id] = []
            candata_scan_obj_idxs[scan_id].append(idx)
            idx += 1

            if self.use_tf_idf:
                obj_nyu_category = self.obj_3D_anno[scan_id][obj_id][2]
                if obj_nyu_category not in n_word_scene[scan_id]:
                    n_word_scene[scan_id][obj_nyu_category] = 0
                n_word_scene[scan_id][obj_nyu_category] += 1
                n_words_per_scene[scan_id] += 1
                if obj_nyu_category not in n_scenes_per_sem:
                    n_scenes_per_sem[obj_nyu_category] = set()
                n_scenes_per_sem[obj_nyu_category].add(scan_id)

        for cand_scan_id in candidate_scans:
            objs_ids_cand_scan = self.scene_graphs[cand_scan_id]["obj_ids"]
            for obj_id in objs_ids_cand_scan:
                obj_3D_idx2info[idx] = self.obj_3D_anno[cand_scan_id][obj_id]
                scans_sg_obj_idxs.append(sg_obj_idxs[cand_scan_id][obj_id])
                cadidate_scans_semantic_ids.append(
                    self.obj_3D_anno[cand_scan_id][obj_id][2]
                )
                if cand_scan_id not in candata_scan_obj_idxs:
                    candata_scan_obj_idxs[cand_scan_id] = []
                candata_scan_obj_idxs[cand_scan_id].append(idx)
                idx += 1

                if self.use_tf_idf:
                    obj_nyu_category = self.obj_3D_anno[cand_scan_id][obj_id][2]
                    if obj_nyu_category not in n_word_scene[cand_scan_id]:
                        n_word_scene[cand_scan_id][obj_nyu_category] = 0
                    n_word_scene[cand_scan_id][obj_nyu_category] += 1
                    n_words_per_scene[cand_scan_id] += 1
                    if obj_nyu_category not in n_scenes_per_sem:
                        n_scenes_per_sem[obj_nyu_category] = set()
                    n_scenes_per_sem[obj_nyu_category].add(cand_scan_id)

            candata_scan_obj_idxs[cand_scan_id] = torch.Tensor(
                candata_scan_obj_idxs[cand_scan_id]
            ).long()
        candata_scan_obj_idxs[scan_id] = torch.Tensor(
            candata_scan_obj_idxs[scan_id]
        ).long()
        scans_sg_obj_idxs = np.array(scans_sg_obj_idxs, dtype=np.int32)
        cadidate_scans_semantic_ids = np.array(
            cadidate_scans_semantic_ids, dtype=np.int32
        )
        scans_sg_obj_idxs = torch.from_numpy(scans_sg_obj_idxs).long()
        cadidate_scans_semantic_ids = torch.from_numpy(
            cadidate_scans_semantic_ids
        ).long()
        if self.use_tf_idf:
            for cand_scan_id in all_candi_scans:
                objs_ids_cand_scan = self.scene_graphs[cand_scan_id]["obj_ids"]
                reweight_matrix_scans[cand_scan_id] = torch.zeros(
                    len(objs_ids_cand_scan)
                )
                obj_idx = 0
                for obj_id in objs_ids_cand_scan:
                    obj_nyu_category = self.obj_3D_anno[cand_scan_id][obj_id][2]
                    n_word_scene_obj = n_word_scene[cand_scan_id][obj_nyu_category]
                    n_words_curr_scene = n_words_per_scene[cand_scan_id]
                    n_scenes_per_sem_obj = len(n_scenes_per_sem[obj_nyu_category])
                    idf = np.log(N_scenes / n_scenes_per_sem_obj)
                    tf_idf = n_word_scene_obj / n_words_curr_scene * idf
                    reweight_matrix_scans[cand_scan_id][obj_idx] = max(tf_idf, 1e-3)
                    obj_idx += 1
        num_objs = idx
        gt_patch_cates = np.zeros(self.num_patch, dtype=np.uint8)
        e1i_matrix = np.zeros((self.num_patch, num_objs), dtype=np.uint8)
        e2j_matrix = np.ones((self.num_patch, num_objs), dtype=np.uint8)
        for patch_h_i in range(self.patch_h):
            patch_h_shift = patch_h_i * self.patch_w
            for patch_w_j in range(self.patch_w):
                patch_idx = patch_h_shift + patch_w_j
                obj_id = gt_2D_anno_flat[patch_idx]
                if obj_id != self.undefined and (obj_id in obj_3D_id2idx_cur_scan):
                    obj_idx = obj_3D_id2idx_cur_scan[obj_id]
                    e1i_matrix[
                        patch_h_shift + patch_w_j, obj_idx
                    ] = 1  # mark 2D-3D patch-object pairs
                    e2j_matrix[
                        patch_h_shift + patch_w_j, obj_idx
                    ] = 0  # mark 2D-3D patch-object unpairs
                    gt_patch_cates[patch_idx] = self.obj_3D_anno[scan_id][obj_id][2]
                else:
                    gt_patch_cates[patch_idx] = self.undefined
        ## e1j_matrix, (num_patch, num_patch), mark unpaired patch-patch pair for image patches
        e1j_matrix = np.zeros((self.num_patch, self.num_patch), dtype=np.uint8)
        for patch_h_i in range(self.patch_h):
            patch_h_shift = patch_h_i * self.patch_w
            for patch_w_j in range(self.patch_w):
                obj_id = gt_2D_anno_flat[patch_h_shift + patch_w_j]
                if obj_id != self.undefined and obj_id in obj_3D_id2idx_cur_scan:
                    e1j_matrix[patch_h_shift + patch_w_j, :] = np.logical_and(
                        gt_2D_anno_flat != self.undefined, gt_2D_anno_flat != obj_id
                    )
                else:
                    e1j_matrix[patch_h_shift + patch_w_j, :] = 1
        ## From 3D to 2D, denote as f1i_matrix, f1j_matrix, f2j_matrix
        ## f1i_matrix = e1i_matrix.T, thus skip
        ## f2j_matrix = e2j_matrix.T, thus skip
        ## f1j_matrix
        obj_cates = [
            obj_3D_idx2info[obj_idx][2] for obj_idx in range(len(obj_3D_idx2info))
        ]
        obj_cates_arr = np.array(obj_cates)
        f1j_matrix = obj_cates_arr.reshape(1, -1) != obj_cates_arr.reshape(-1, 1)

        assoc_data_dict = {
            "e1i_matrix": torch.from_numpy(e1i_matrix).float(),
            "e1j_matrix": torch.from_numpy(e1j_matrix).float(),
            "e2j_matrix": torch.from_numpy(e2j_matrix).float(),
            "f1j_matrix": torch.from_numpy(f1j_matrix).float(),
            "gt_patch_cates": gt_patch_cates,
            "scans_sg_obj_idxs": scans_sg_obj_idxs,
            "cadidate_scans_semantic_ids": cadidate_scans_semantic_ids,
            "candata_scan_obj_idxs": candata_scan_obj_idxs,
            "reweight_matrix_scans": reweight_matrix_scans,
            "n_scenes_per_sem": n_scenes_per_sem,
        }
        return assoc_data_dict

    def _aggregate(
        self, data_dict: list, key: str, mode: str
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
            return sparse_cat([data[key] for data in data_dict])
        else:
            raise NotImplementedError

    def _collate(self, batch: list) -> dict:
        scans_batch = [data["scan_id"] for data in batch]

        if self.use_cross_scene:
            if self.split == "train":
                candidate_scans, union_scans = self.sample_candidate_scenes_for_scans(
                    scans_batch, self.num_scenes
                )
            else:
                candidate_scans = {}
                for scan_id in scans_batch:
                    candidate_scans[scan_id] = self.candidate_scans[scan_id]
                union_scans = list(
                    set(
                        scans_batch
                        + [
                            scan
                            for scan_list in candidate_scans.values()
                            for scan in scan_list
                        ]
                    )
                )
        else:
            candidate_scans, union_scans = None, scans_batch

        batch_size = len(batch)
        data_dict = {}
        data_dict["batch_size"] = batch_size
        data_dict["temporal"] = self.temporal
        data_dict["scan_ids"] = np.stack([data["scan_id"] for data in batch])
        if self.temporal:
            data_dict["scan_ids_temp"] = np.stack(
                [data["scan_id_temporal"] for data in batch]
            )
        data_dict["frame_idxs"] = np.stack([data["frame_idx"] for data in batch])
        if self.use_2D_feature:
            patch_features_batch = np.stack(
                [data["patch_features"] for data in batch]
            )  # (B, P_H, P_W, D)
            data_dict["patch_features"] = torch.from_numpy(
                patch_features_batch
            ).float()  # (B, H, W, C)
        else:
            images_batch = np.stack([data["image"] for data in batch])
            data_dict["images"] = torch.from_numpy(images_batch).float()  # (B, H, W, C)
            if self.cfg.data.img_encoding.record_feature:
                data_dict["patch_features_paths"] = [
                    data["patch_features_path"] for data in batch
                ]
        data_dict["obj_2D_patch_anno_flatten_list"] = [
            torch.from_numpy(data["obj_2D_patch_anno_flatten"]) for data in batch
        ]  # B - [N_P]
        if self.temporal:
            scene_graph_scans = list(
                set(union_scans + [data["scan_id_temporal"] for data in batch])
            )
        else:
            scene_graph_scans = union_scans
        scene_graph_infos = [
            self.scene_graphs[scan_id] for scan_id in scene_graph_scans
        ]
        scene_graphs_ = {}
        scans_size = len(scene_graph_infos)
        scene_graphs_["batch_size"] = scans_size
        scene_graphs_["obj_ids"] = self._aggregate(
            scene_graph_infos, "obj_ids", "np_concat"
        )
        splat_dict = [
            self._load_splats(scan_id, preload_slat=self.cfg.data.preload_slat)
            for scan_id in scene_graph_scans
        ]
        scene_graphs_["tot_obj_pts"] = self._aggregate(
            scene_graph_infos, "tot_obj_pts", "torch_cat"
        )
        scene_graphs_["tot_obj_splat"] = self._aggregate(
            splat_dict, "tot_obj_splat", "sparse_cat"
        ).float()
        assert (
            scene_graphs_["tot_obj_splat"].shape[0]
            == scene_graphs_["tot_obj_pts"].shape[0]
        ), f"{scene_graphs_['tot_obj_splat'].shape[0]} != {scene_graphs_['tot_obj_pts'].shape[0]}"
        scene_graphs_["mean_obj_splat"] = self._aggregate(
            splat_dict, "mean_obj_splat", "torch_cat"
        ).float()
        scene_graphs_["scale_obj_splat"] = self._aggregate(
            splat_dict, "scale_obj_splat", "torch_cat"
        ).float()
        scene_graphs_["graph_per_obj_count"] = self._aggregate(
            scene_graph_infos, "graph_per_obj_count", "np_stack"
        )
        scene_graphs_["graph_per_edge_count"] = self._aggregate(
            scene_graph_infos, "graph_per_edge_count", "np_stack"
        )
        scene_graphs_["tot_obj_count"] = self._aggregate(
            scene_graph_infos, "tot_obj_count", "np_stack"
        )
        scene_graphs_["tot_bow_vec_object_attr_feats"] = self._aggregate(
            scene_graph_infos, "tot_bow_vec_object_attr_feats", "torch_cat"
        ).double()
        scene_graphs_["tot_bow_vec_object_edge_feats"] = self._aggregate(
            scene_graph_infos, "tot_bow_vec_object_edge_feats", "torch_cat"
        ).double()
        scene_graphs_["tot_rel_pose"] = self._aggregate(
            scene_graph_infos, "tot_rel_pose", "torch_cat"
        ).double()
        scene_graphs_["edges"] = self._aggregate(
            scene_graph_infos, "edges", "torch_cat"
        )
        scene_graphs_["global_obj_ids"] = self._aggregate(
            scene_graph_infos, "global_obj_ids", "np_concat"
        )
        scene_graphs_["scene_ids"] = self._aggregate(
            scene_graph_infos, "scene_ids", "np_stack"
        )
        scene_graphs_["pcl_center"] = self._aggregate(
            scene_graph_infos, "pcl_center", "np_stack"
        )
        if self.use_aug_3D and self.split == "train":
            num_obs = scene_graphs_["tot_obj_pts"].shape[1]
            pcs_flatten = scene_graphs_["tot_obj_pts"].reshape(-1, 3)
            pcs_distorted_flatten = self.elastic_distortion(pcs_flatten)
            scene_graphs_["tot_obj_pts"] = pcs_distorted_flatten.reshape(-1, num_obs, 3)
        if "voxel" in self.sgaligner_modules or "img_patch" in self.sgaligner_modules:
            obj_img_patches = {}
            obj_img_poses = {}
            obj_intrinsics = {}
            obj_img_top_frames = {}
            obj_count_ = 0
            obj_annos = defaultdict(dict)
            for scan_idx, scan_id in enumerate(scene_graphs_["scene_ids"]):
                scan_id = scan_id[0]

                obj_start_idx, obj_end_idx = (
                    obj_count_,
                    obj_count_ + scene_graphs_["tot_obj_count"][scan_idx],
                )
                obj_ids = scene_graphs_["obj_ids"][obj_start_idx:obj_end_idx]
                obj_img_patches_scan_tops = self.obj_img_patches_scan_tops[scan_id]
                obj_img_patches_scan = obj_img_patches_scan_tops["obj_visual_emb"]
                obj_top_frames = obj_img_patches_scan_tops["obj_image_votes_topK"]

                obj_img_patches[scan_id] = {}
                obj_img_poses[scan_id] = {}
                for obj_id in obj_ids:
                    if obj_id not in obj_top_frames:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                        obj_img_patches[scan_id][obj_id] = torch.from_numpy(
                            obj_img_patch_embs
                        ).float()
                        if self.use_pos_enc:
                            identity_pos = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(
                                1, -1
                            )
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
                                obj_img_poses_list.append(
                                    self.image_poses[scan_id][frame_idx]
                                )

                    if len(obj_img_patch_embs_list) == 0:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                        if self.use_pos_enc:
                            obj_img_poses_arr = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(
                                1, -1
                            )
                    else:
                        obj_img_patch_embs = np.concatenate(
                            obj_img_patch_embs_list, axis=0
                        )
                        if self.use_pos_enc:
                            obj_img_poses_arr = np.stack(obj_img_poses_list, axis=0)

                    obj_img_patches[scan_id][obj_id] = torch.from_numpy(
                        obj_img_patch_embs
                    ).float()

                    obj_intrinsics[scan_id] = self.image_intrinsics[scan_id]
                    obj_img_top_frames[scan_id] = obj_top_frames

                    if self.preload_masks:
                        scan_annos = common.load_pkl_data(
                            self.obj_2D_annos_pathes[scan_id]
                        )
                        for frames in obj_top_frames.values():
                            for frame in frames:
                                obj_annos[scan_id][frame] = scan_annos[frame]
                        scene_graphs_["obj_annos"] = obj_annos

                    if self.use_pos_enc:
                        obj_img_poses[scan_id][obj_id] = torch.from_numpy(
                            obj_img_poses_arr
                        ).float()
                obj_count_ += scene_graphs_["tot_obj_count"][scan_idx]

            scene_graphs_["obj_img_patches"] = obj_img_patches
            scene_graphs_["obj_img_top_frames"] = obj_img_top_frames
            scene_graphs_["obj_intrinsics"] = obj_intrinsics
            if self.use_pos_enc:
                scene_graphs_["obj_img_poses"] = obj_img_poses

        if self.split == "test":
            obj_train_frames = defaultdict(dict)
            obj_test_frames = defaultdict(dict)
            obj_train_poses = defaultdict(dict)
            obj_test_poses = defaultdict(dict)
            obj_intrinsics = {}

            for scan_id in scene_graphs_["scene_ids"]:
                scan_id = scan_id[0]
                train_frames = set(
                    [
                        frame
                        for frames in self.train_frames[scan_id].values()
                        for frame in frames
                    ]
                )
                test_frames = sorted(
                    list(
                        set(
                            [
                                frame
                                for frames in self.test_frames[scan_id].values()
                                for frame in frames
                            ]
                        )
                        - train_frames
                    )
                )
                train_frames = sorted(list(train_frames))
                if len(train_frames) == 0 or len(test_frames) == 0:
                    print(f"{scan_id} has no train or test frames")
                    continue
                train_poses = [
                    self.image_poses[scan_id][frame] for frame in train_frames
                ]
                test_poses = [self.image_poses[scan_id][frame] for frame in test_frames]
                train_poses = torch.from_numpy(np.stack(train_poses)).float()
                test_poses = torch.from_numpy(np.stack(test_poses)).float()

                obj_test_frames[scan_id] = test_frames
                obj_train_frames[scan_id] = train_frames
                obj_train_poses[scan_id] = train_poses
                obj_test_poses[scan_id] = test_poses
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

        assoc_data_dict, assoc_data_dict_temporal = [], []
        sg_obj_idxs = {}
        sg_obj_idxs_tensor = {}
        sg_obj_idx_start = 0
        for scan_idx, scan_id in enumerate(scene_graphs_["scene_ids"]):
            scan_id = scan_id[0]
            sg_obj_idxs[scan_id] = {}
            objs_count = scene_graphs_["tot_obj_count"][scan_idx]
            sg_obj_idxs_tensor[scan_id] = torch.from_numpy(
                np.arange(sg_obj_idx_start, sg_obj_idx_start + objs_count)
            ).long()
            for sg_obj_idx in range(sg_obj_idx_start, sg_obj_idx_start + objs_count):
                obj_id = scene_graphs_["obj_ids"][sg_obj_idx]
                sg_obj_idxs[scan_id][obj_id] = sg_obj_idx
            sg_obj_idx_start += objs_count
        for data in batch:
            (
                assoc_data_dict_curr,
                assoc_data_dict_temporal_curr,
            ) = self._get_obj_patch_assoc_dict(data, candidate_scans, sg_obj_idxs)
            assoc_data_dict.append(assoc_data_dict_curr)
            assoc_data_dict_temporal.append(assoc_data_dict_temporal_curr)
        data_dict["assoc_data_dict"] = assoc_data_dict
        data_dict["assoc_data_dict_temp"] = assoc_data_dict_temporal
        data_dict["sg_obj_idxs"] = sg_obj_idxs
        data_dict["sg_obj_idxs_tensor"] = sg_obj_idxs_tensor
        data_dict["candidate_scans"] = candidate_scans
        if len(batch) > 0:
            return data_dict
        else:
            return None

    def __getitem__(self, idx: int) -> dict:
        data_item = self.data_items[idx]
        data_dict = self._item_to_dict(data_item, self.temporal)
        return data_dict

    def collate_fn(self, batch: list) -> dict:
        return self._collate(batch)

    def __len__(self) -> int:
        return len(self.data_items)
