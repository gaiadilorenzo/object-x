import json
import logging
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scannet

_LOGGER = logging.getLogger(__name__)


def select_train_test_frames(
    frame_idxs: List[int],
    extrinsics: Dict[int, np.ndarray],
    masks: Dict[int, np.ndarray],
) -> Tuple[List[int], List[int]]:
    """
    Clusters frames by pose into 4 clusters and selects one representative per cluster for training,
    choosing the frame that minimizes the number of masked points.

    Args:
        frame_idxs: List of frame indices.
        extrinsics: Dictionary of frame extrinsics.
        masks: Dictionary of frame masks.

    Returns:
        Tuple[List[int], List[int]]: Train and test frame indices.
    """

    if len(frame_idxs) <= 5:
        return frame_idxs, []
    viewpoints = np.array([extrinsics[frame_id][:3, 3] for frame_id in frame_idxs])
    num_clusters = min(4, len(viewpoints))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(viewpoints)
    cluster_centers = kmeans.cluster_centers_
    train_indices = []

    selected_frames = set()
    for center in cluster_centers:
        distances = np.linalg.norm(viewpoints - center, axis=1)
        cluster_frames = np.argsort(distances)[:20]  # Take top closest frames
        best_frame = min(
            cluster_frames,
            key=lambda i: np.sum(masks[frame_idxs[i]] == 0)
            if i not in selected_frames
            else float("inf"),
        )
        train_indices.append(best_frame)
        selected_frames.add(best_frame)
    test_indices = list(set(range(len(frame_idxs))) - set(train_indices))
    return [frame_idxs[i] for i in train_indices], [frame_idxs[i] for i in test_indices]


def process_data(cfg: Config, split: str = "train"):
    """
    Process data to generate train/test splits for 3RScan and save to file.

    Args:
        cfg (Config): Config object.
        split (str): Split.
    """

    root_dir = cfg.data.root_dir
    subscan_ids_generated = np.genfromtxt(
        osp.join(cfg.data.root_dir, "files", "scannet_test_split.txt"),
        dtype=str,
    )
    all_subscan_ids = subscan_ids_generated

    train_test_splits = {}
    output_path = osp.join(
        root_dir, "files", f"{split}_train_test_splits_scannet_tmppppp.json"
    )

    for scan_id in tqdm(all_subscan_ids):
        if not osp.exists(
            osp.join(
                cfg.data.root_dir,
                "scene_graph_fusion",
                scan_id,
                "{}.pkl".format(scan_id),
            )
        ):
            _LOGGER.warning(f"Skipping {scan_id} as scene graph is not available")
            continue

        scene_graph_dict = common.load_pkl_data(
            osp.join(
                cfg.data.root_dir,
                "scene_graph_fusion",
                scan_id,
                "{}.pkl".format(scan_id),
            )
        )
        obj_data = scene_graph_dict["objects_id"]
        train_test_splits[scan_id] = {}
        for obj_id in obj_data:
            frame_idxs, masks = scannet.load_frame_idxs_per_obj(
                data_dir=cfg.data.root_dir, scan_id=scan_id, obj_id=obj_id
            )
            masks = {frame_idx: mask for frame_idx, mask in zip(frame_idxs, masks)}
            extrinsics = scannet.load_frame_poses(
                data_split_dir=cfg.data.root_dir, scan_id=scan_id, skip=25
            )
            train_frames, test_frames = select_train_test_frames(
                frame_idxs, extrinsics, masks
            )
            train_test_splits[scan_id][str(obj_id)] = {
                "train": train_frames,
                "test": test_frames,
            }

        with open(output_path, "w") as f:
            json.dump(train_test_splits, f, indent=4)

    with open(output_path, "w") as f:
        json.dump(train_test_splits, f, indent=4)
    _LOGGER.info(f"Saved train/test splits to {output_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, default="test")
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    common.init_log(level=logging.INFO)
    _LOGGER.info("***** Generate train/test splits for ScanNet *****")
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    process_data(cfg, split=args.split)
