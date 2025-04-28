import logging
import os.path as osp
import sys
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import numpy as np
import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from plyfile import PlyElementParseError
from render import render_sets
from tqdm import tqdm
from train import training

from configs import Config, update_configs
from utils import common
from utils.gaussian_splatting import GaussianSplat
from utils.general_utils import safe_state

_LOGGER = logging.getLogger(__name__)


def annotate_gaussian(
    obj_data: Dict[str, str],
    scan_id: str,
    iteration: int,
    mode: str = "gs_annotations",
) -> None:
    """
    Annotate gaussian splats for the specified scan.

    Args:
        obj_data (Dict[str, str]): Object data.
        scan_id (str): Scan ID.
        iteration (int): Iteration number.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
    """

    for i, obj in enumerate(obj_data["objects"]):
        try:
            obj_id = int(obj["id"])
            optim_params = op.extract(args)
            args.model_path = osp.join(
                args.model_dir, "files", mode, scan_id, str(obj_id)
            )
            gaussian_splat_ply_file = osp.join(
                args.model_path,
                "point_cloud",
                f"iteration_{optim_params.iterations}",
                "point_cloud.ply",
            )

            try:
                if osp.exists(gaussian_splat_ply_file) and not args.overwrite:
                    _LOGGER.debug(f"Splat found. Loading {scan_id} ({obj_id}).")
                    GaussianSplat.load_ply(gaussian_splat_ply_file)
                    continue
            except (PlyElementParseError) as e:
                _LOGGER.debug(f"Error loading {scan_id} ({obj_id}): {e}.")
                pass

            args.object_id = obj_id
            args.name = "Scan3R"
            args.source_path = osp.join(args.source_dir, "files", mode, scan_id)

            torch.cuda.empty_cache()
            safe_state(args.quiet)
            torch.autograd.set_detect_anomaly(args.detect_anomaly)
            training(
                lp.extract(args),
                op.extract(args),
                pp.extract(args),
                args.test_iterations,
                args.save_iterations,
                args.checkpoint_iterations,
                args.start_checkpoint,
                args.debug_from,
            )
            if (
                iteration % args.render_iterations == 0
                and i % args.render_iterations == 0
            ):
                render_sets(
                    lp.extract(args),
                    args.iterations,
                    pp.extract(args),
                    args.skip_train,
                    skip_test=True,
                )

        except FileNotFoundError as e:
            _LOGGER.exception(f"Error processing {scan_id} ({obj_id}): {e}")


def process_data(
    cfg: Config, mode: str = "gs_annotations", split: str = "train"
) -> np.ndarray:
    """
    Process subscans from the specified split for gaussian annotations.

    Args:
        cfg: Configuration object.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
        split (str, optional): Split to run subscan generation on. Defaults to "train".

    Returns:
        np.ndarray: processed subscan IDs.
    """

    scan_type = cfg.autoencoder.encoder.scan_type
    resplit = "resplit_" if cfg.data.resplit else ""
    scan_ids_filename = (
        f"{split}_{resplit}scans.txt"
        if scan_type == "scan"
        else f"{split}_scans_subscenes.txt"
    )
    objects_info_file = osp.join(root_dir, "files", "objects.json")
    all_obj_info = common.load_json(objects_info_file)

    subscan_ids_generated = np.genfromtxt(
        osp.join(root_dir, "files", scan_ids_filename), dtype=str
    )
    subscan_ids_processed = []

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

    pbar = tqdm(all_subscan_ids)
    for iter_, subscan_id in enumerate(pbar):
        pbar.set_description(f"Processing {subscan_id}")
        obj_data = next(
            obj_data
            for obj_data in all_obj_info["scans"]
            if obj_data["scan"] == subscan_id
        )

        try:
            annotate_gaussian(
                mode=mode,
                obj_data=obj_data,
                scan_id=subscan_id,
                iteration=iter_,
            )
        except Exception as e:
            _LOGGER.exception(f"Error processing {subscan_id}: {e}")
            _LOGGER.error(f"Skipping {subscan_id}")
            continue

        subscan_ids_processed.append(subscan_id)

    subscan_ids = np.array(subscan_ids_processed)
    return subscan_ids


def parse_args() -> Tuple[
    Namespace, ModelParams, OptimizationParams, PipelineParams, list[str]
]:
    """
    Parse command line arguments.

    Returns:
        Tuple[argparse.Namespace, ModelParams, OptimizationParams, PipelineParams, List[str]]: Parsed arguments, 3DGS args and unknown arguments.
    """

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--config",
        dest="config",
        type=str,
    )
    parser.add_argument(
        "--split",
        dest="split",
        default="train",
        type=str,
    )
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render_iterations", type=int, default=1000)
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--source_dir", type=str, default="")
    args, unknown = parser.parse_known_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    return args, lp, op, pp, unknown


if __name__ == "__main__":
    common.init_log(level=logging.DEBUG)
    _LOGGER.info("**** Starting annotation of gaussian splats for 3RScan ****")
    args, lp, op, pp, unknown = parse_args()
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    root_dir = cfg.data.root_dir
    scan_ids = process_data(cfg, mode="gs_annotations", split=args.split)
