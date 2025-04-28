import json
import logging
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scan3r

_LOGGER = logging.getLogger(__name__)


def select_train_test_frames(
    frame_idxs: List[int],
    extrinsics: Dict[int, np.ndarray],
    masks: Dict[int, np.ndarray],
) -> Tuple[List[int], List[int]]:
    """
    Clusters frames by pose into 12 clusters and selects one representative per cluster for training,
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
    num_clusters = min(12, len(viewpoints))
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
    scenes_dir = osp.join(root_dir, "scenes")
    scan_ids_filename = f"{split}_resplit_scans.txt"
    scans_files_dir = osp.join(root_dir, "files")
    all_scan_data = common.load_json(osp.join(scans_files_dir, "3RScan.json"))

    subscan_ids_generated = np.genfromtxt(
        osp.join(root_dir, "files", scan_ids_filename), dtype=str
    )

    subRescan_ids_generated = {}
    scans_dir = cfg.data.root_dir
    scans_files_dir = osp.join(scans_dir, "files")

    all_scan_data = common.load_json(osp.join(scans_files_dir, "3RScan.json"))

    for scan_data in all_scan_data:
        ref_scan_id = scan_data["reference"]
        if ref_scan_id in subscan_ids_generated:
            rescan_ids = [scan["reference"] for scan in scan_data["scans"]]
            subRescan_ids_generated[ref_scan_id] = [ref_scan_id] + rescan_ids

    subscan_ids_generated = subRescan_ids_generated

    all_subscan_ids = [
        subscan_id
        for scan_id in subscan_ids_generated
        for subscan_id in subscan_ids_generated[scan_id]
    ]
    all_obj_info = common.load_json(osp.join(root_dir, "files", "objects.json"))
    train_test_splits = {}
    output_path = osp.join(root_dir, "files", f"{split}_train_test_splits.json")

    for scan_id in tqdm(all_subscan_ids):
        obj_data = next(obj for obj in all_obj_info["scans"] if obj["scan"] == scan_id)

        train_test_splits[scan_id] = {}
        for obj in obj_data["objects"]:
            obj_id = str(obj["id"])
            frame_idxs, masks = scan3r.load_frame_idxs_per_obj(
                data_dir=root_dir, scan_id=scan_id, obj_id=obj_id
            )
            masks = {frame_idx: mask for frame_idx, mask in zip(frame_idxs, masks)}
            extrinsics = scan3r.load_frame_poses(
                data_dir=scenes_dir, scan_id=scan_id, frame_idxs=frame_idxs
            )
            train_frames, test_frames = select_train_test_frames(
                frame_idxs, extrinsics, masks
            )
            train_test_splits[scan_id][obj_id] = {
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
    _LOGGER.info("***** Generate train/test splits for 3RScan *****")
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    process_data(cfg, split=args.split)
