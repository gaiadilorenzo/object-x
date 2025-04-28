import argparse
import logging
import os.path as osp

import numpy as np
from tqdm.auto import tqdm

from configs import Config, update_configs
from utils import common, scannet

SCENE_GRAPH_FUSION_EXE = "dependencies/SceneGraphFusion/bin/exe_GraphSLAM"
SCENE_GRAPH_FUSION_MODEL = "dependencies/SCENE-GRAPH-FUSION"

_LOGGER = logging.getLogger(__name__)


class ScannetSGPrediction:
    def __init__(self, cfg: Config, num_jobs: int, out_dir: str, split: str = "test"):
        self.cfg = cfg
        self.split = split
        self.out_dir = out_dir
        self.parallel_jobs = num_jobs

        # Load scan information
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

    def run(self):
        commands = []

        for scan_id in tqdm(self.scan_ids, desc="Processing scans"):
            scan_folder = osp.join(self.cfg.data.root_dir, "scans", scan_id)
            seq_file = osp.join(scan_folder, f"{scan_id}.sens")
            out_folder = osp.join(scan_folder, "scene_graph_fusion")

            exe_command = f"{SCENE_GRAPH_FUSION_EXE} --pth_in {seq_file} --pth_out {out_folder} --pth_model {SCENE_GRAPH_FUSION_MODEL}"
            commands.append(exe_command)

        # Run commands in parallel
        scannet.run_bash_batch(commands, jobs_per_step=self.parallel_jobs)

        # Verify processing
        self.verify_and_rerun()

    def verify_and_rerun(self):
        all_processed = False
        while not all_processed:
            commands = []

            for scan_id in self.scan_ids:
                out_folder = osp.join(self.out_dir, scan_id, "scene_graph_fusion")
                files_to_check = ["predictions.json", "inseg.ply", "node_semantic.ply"]

                if not all(
                    osp.exists(osp.join(out_folder, file)) for file in files_to_check
                ):
                    scan_folder = osp.join(self.out_dir, scan_id)
                    seq_file = osp.join(scan_folder, f"{scan_id}.sens")
                    exe_command = f"{SCENE_GRAPH_FUSION_EXE} --pth_in {seq_file} --pth_out {out_folder} --pth_model {SCENE_GRAPH_FUSION_MODEL}"
                    commands.append(exe_command)

            all_processed = not commands
            if commands:
                scannet.run_bash_batch(commands, jobs_per_step=self.parallel_jobs)


def parse_args():
    parser = argparse.ArgumentParser(description="Scene Graph Prediction for ScanNet")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for scene graph predictions",
    )
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
    _LOGGER.info("***** Scene Graph Prediction for ScanNet *****")
    args, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    predictor = ScannetSGPrediction(
        cfg, num_jobs=args.num_jobs, split="test", out_dir=args.out_dir
    )
    predictor.run()


if __name__ == "__main__":
    main()
