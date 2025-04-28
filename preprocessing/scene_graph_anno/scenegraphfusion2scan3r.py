import argparse
import logging
import os.path as osp

import numpy as np
from tqdm.auto import tqdm

from configs import Config, update_configs
from utils import common, scannet

_LOGGER = logging.getLogger(__name__)


class ScannetSGPrediction:
    def __init__(self, cfg: Config, split: str = "test"):
        self.cfg = cfg

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

        self.pred_folders = {}
        for scan_id in self.scan_ids:
            scan_folder = osp.join(cfg.data.root_dir, "scenes", scan_id)
            self.pred_folders[scan_id] = osp.join(
                cfg.data.root_dir, "scene_graph_fusion", scan_id
            )

        rel_class_file = osp.join(
            cfg.data.root_dir, "files", "scannet8_relationships.txt"
        )
        self.rel2idx = common.name2idx(rel_class_file)
        obj_class_file = osp.join(cfg.data.root_dir, "files", "scannet20_classes.txt")
        self.class2idx = common.name2idx(obj_class_file)

    def run(self):
        """Convert Scene Graph Prediction for ScanNet to 3RScan."""

        for scan_id in tqdm(self.scan_ids):
            pred_folder = self.pred_folders[scan_id]
            if not osp.exists(pred_folder):
                print(f"File not found {pred_folder}")
                continue
            data_dict = scannet.scenegraphfusion2scan3r(
                scan_id, pred_folder, self.rel2idx, self.class2idx, self.cfg
            )

            file = osp.join(pred_folder, "{}.pkl".format(scan_id))
            common.write_pkl_data(data_dict, file)
            scannet.calculate_bow_node_edge_feats(file, self.rel2idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Scene Graph Prediction for ScanNet")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default="preprocessing/scene_graph_anno/sg_anno_scannet.yaml",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1, help="Number of parallel jobs to run"
    )
    return parser.parse_known_args()


def main():
    common.init_log()
    _LOGGER.info("***** Convert Scene Graph Prediction for ScanNet to 3RScan *****")
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    scannet_dino_generator = ScannetSGPrediction(cfg, split="test")
    scannet_dino_generator.run()


if __name__ == "__main__":
    main()
