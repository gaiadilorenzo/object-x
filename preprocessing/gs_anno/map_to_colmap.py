import argparse
import logging
import os
import os.path as osp
import shutil
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

from configs import Config, update_configs
from utils import common, scan3r
from utils.geometry import rotation_matrix_to_quaternion
from utils.visualisation import save_frustum

_LOGGER = logging.getLogger(__name__)


def write_cameras_txt(intrinsics: Dict[str, np.ndarray], scan_id: str) -> None:
    """
    Write camera intrinsics to COLMAP format.

    Args:
        intrinsics (Dict[str, np.ndarray]): Dictionary with keys: fx, fy, cx, cy (focal lengths and principal point).
        scan_id (str): Scan identifier.
    """

    cameras_dir = os.path.join(output_dir, scan_id, "sparse", "0")
    os.makedirs(cameras_dir, exist_ok=True)

    with open(os.path.join(cameras_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(
            f"1 PINHOLE {intrinsics['width']} {intrinsics['height']} "
            f"{intrinsics['intrinsic_mat'][0,0]} {intrinsics['intrinsic_mat'][1,1]} "
            f"{intrinsics['intrinsic_mat'][0, 2]} {intrinsics['intrinsic_mat'][1, 2]}\n"
        )


def write_images_txt(
    frame_poses: Dict[str, np.ndarray],
    frame_paths: List[str],
    scan_id: str,
) -> None:
    """
    Write camera extrinsics (rotation and translation) to COLMAP format (images.txt).

    Note:
        The reconstructed pose of an image is specified as the projection from world to the camera coordinate system
        of an image using a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ). The quaternion is defined
        using the Hamilton convention, which is, for example, also used by the Eigen library. The coordinates of the
        projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of the 3x3 rotation matrix
        composed from the quaternion and T is the translation vector. The local camera coordinate system of an image is
        defined in a way that the X axis points to the right, the Y axis to the bottom, and the Z axis to the front as
        seen from the image.

    Args:
        frame_poses (Dict[str, np.ndarray]): List of 4x4 transformation matrices (extrinsics).
        frame_paths (List[str]): List of paths to the images.
        scan_id (str): Scan identifier.
    """

    images_dir = os.path.join(output_dir, scan_id, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    with open(os.path.join(images_dir, "images.txt"), "w") as f:
        f.write("# Image list with one line of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_PATH\n")

        for idx, path in enumerate(frame_paths):
            pose = frame_poses[path]
            pose = np.linalg.inv(pose)
            R = pose[:3, :3]
            t = pose[:3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
            tx, ty, tz = t
            full_path = osp.join(
                root_dir, "scenes", scan_id, "sequence", f"frame-{path}.color.jpg"
            )
            f.write(f"{idx+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {full_path}\n")
            f.write("0.0 0.0 -1\n")


def write_points3D_txt(
    obj_file: str,
    scan_id: str,
) -> None:
    """
    Convert point cloud from .ply file to COLMAP format (points3D.txt).

    Args:
        obj_file (str): Path to the point cloud .ply file.
        scan_id (str): Scan identifier.
    """

    pcd = o3d.io.read_point_cloud(obj_file)
    points = pcd.points
    colors = pcd.colors if pcd.has_colors() else np.zeros_like(points)
    pointcloud_dir = os.path.join(output_dir, scan_id, "sparse", "0")

    os.makedirs(pointcloud_dir, exist_ok=True)
    with open(os.path.join(pointcloud_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (point, color) in enumerate(zip(points, colors)):
            x, y, z = point
            r, g, b = (color * 255).astype(int)
            f.write(f"{i+1} {x} {y} {z} {r} {g} {b} 0.0\n")


def _write_camera_frustum_ply(
    frame_poses: Dict[str, np.ndarray],
    frame_paths: List[str],
    intrinsics_mat: Dict[str, np.ndarray],
    scan_id: str,
) -> None:

    width = intrinsics_mat["width"]
    height = intrinsics_mat["height"]
    intrinsics = intrinsics_mat["intrinsic_mat"]

    for path in frame_paths[::50]:
        pose = frame_poses[path]
        R = pose[:3, :3]
        t = pose[:3, 3]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        save_frustum(
            os.path.join(args.vis_dir, f"{scan_id}_camera_frustum_{path}.ply"),
            R=R,
            t=t,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )


def _render_random_view(
    scan_id: str,
    frame_poses: Dict[str, np.ndarray],
    frame_paths: List[str],
    intrinsics: Dict[str, np.ndarray],
    num_views: int = 20,
) -> None:

    skip = len(frame_paths) // num_views
    for frame_idx in frame_paths[::skip]:
        pose = np.linalg.inv(frame_poses[frame_idx])
        R = pose[:3, :3]
        t = pose[:3, 3]
        fx, fy = float(intrinsics["intrinsic_mat"][0, 0]), float(
            intrinsics["intrinsic_mat"][1, 1]
        )
        cx, cy = float(intrinsics["intrinsic_mat"][0, 2]), float(
            intrinsics["intrinsic_mat"][1, 2]
        )
        _render(
            int(intrinsics["width"]),
            int(intrinsics["height"]),
            fx,
            fy,
            cx,
            cy,
            R,
            t,
            f"{scan_id}_{frame_idx}",
            scan_id,
        )


def _render(width, height, fx, fy, cx, cy, R, t, image_name, scan_id):
    mesh = o3d.io.read_triangle_mesh(
        osp.join(root_dir, "scenes", scan_id, "labels.instances.annotated.v2.ply")
    )
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
    material = o3d.visualization.rendering.MaterialRecord()
    renderer.scene.add_geometry("mesh", mesh, material)

    extrinsics_matrix = np.eye(4)
    extrinsics_matrix[:3, :3] = R
    extrinsics_matrix[:3, 3] = t
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    renderer.setup_camera(intrinsics, extrinsics_matrix)

    image = renderer.render_to_image()
    o3d.io.write_image(osp.join(args.vis_dir, f"{image_name}.png"), image)


def map_to_colmap(data_dir: str, scan_id: str) -> None:
    """
    Convert data to COLMAP format.

    Note: if visualize flag is set, camera frustums, point cloud and random views are saved to the visualization directory.

    Args:
        data_dir (str): Directory containing the data.
        scan_id (str): Scan identifier.
    """

    mesh_file = osp.join(
        data_dir, "scenes", scan_id, "labels.instances.annotated.v2.ply"
    )
    intrinsics = scan3r.load_intrinsics(osp.join(data_dir, "scenes"), scan_id)
    frame_idxs = scan3r.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id)
    image_poses = scan3r.load_frame_poses(
        osp.join(data_dir, "scenes"), scan_id, frame_idxs=frame_idxs
    )
    write_cameras_txt(intrinsics, scan_id)
    write_images_txt(image_poses, frame_idxs, scan_id)
    write_points3D_txt(mesh_file, scan_id)

    if args.visualize:
        _visualize(scan_id, mesh_file, intrinsics, frame_idxs, image_poses)


def _visualize(
    scan_id: str,
    ply_data_npy_file: str,
    intrinsics: Dict[str, np.ndarray],
    frame_idxs: List[str],
    image_poses: Dict[str, np.ndarray],
) -> None:
    shutil.copy(
        ply_data_npy_file,
        os.path.join(args.vis_dir, f"{scan_id}_points3D_original.ply"),
    )
    _write_camera_frustum_ply(image_poses, frame_idxs, intrinsics, scan_id)
    _render_random_view(
        scan_id, image_poses, frame_idxs, intrinsics, num_views=args.num_views
    )


def process_subscan(subscan_id, data_dir):
    _LOGGER.info(f"Processing subscan {subscan_id}")
    if map_to_colmap(data_dir, scan_id=subscan_id) != -1:
        _LOGGER.info(f"Done processing subscan {subscan_id}")
        return subscan_id
    return None


def process_data(cfg: Config, split: str = "train") -> np.ndarray:
    """
    Process subscans from the specified split to generate COLMAP data.

    Args:
        cfg: Configuration object.
        split (str, optional): Split to run subscan generation on. Defaults to "train".

    Returns:
        np.ndarray: processed subscan IDs.
    """

    use_predicted = cfg.autoencoder.encoder.use_predicted
    scan_type = cfg.autoencoder.encoder.scan_type
    out_dirname = "" if scan_type == "scan" else "out"
    out_dirname = osp.join(out_dirname, "predicted") if use_predicted else out_dirname
    data_dir = osp.join(cfg.data.root_dir, out_dirname)

    resplit = "resplit_" if cfg.data.resplit else ""
    scan_ids_filename = (
        f"{split}_{resplit}scans.txt"
        if scan_type == "scan"
        else f"{split}_scans_subscenes.txt"
    )

    subscan_ids_generated = np.genfromtxt(
        osp.join(data_dir, "files", scan_ids_filename), dtype=str
    )
    subRescan_ids_generated = {}
    scans_dir = cfg.data.root_dir
    scans_files_dir = osp.join(scans_dir, "files")

    all_scan_data = common.load_json(osp.join(scans_files_dir, "3RScan.json"))

    for scan_data in all_scan_data:
        ref_scan_id = scan_data["reference"]

        if ref_scan_id in subscan_ids_generated:
            rescan_ids = [scan["reference"] for scan in scan_data["scans"]]
            subRescan_ids_generated[ref_scan_id] = rescan_ids + [ref_scan_id]

    subscan_ids_generated = subRescan_ids_generated
    all_subscan_ids = [
        subscan_id
        for scan_id in subscan_ids_generated
        for subscan_id in subscan_ids_generated[scan_id]
    ]

    with Pool(args.num_workers) as pool:
        results = pool.starmap(
            process_subscan,
            [(subscan_id, data_dir) for subscan_id in all_subscan_ids],
        )

    subscan_ids_processed = [res for res in results if res is not None]
    subscan_ids = np.array(subscan_ids_processed)
    return subscan_ids


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse command line arguments.

    Returns:
        Tuple[argparse.Namespace, List[str]]: Parsed arguments and unknown arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        default="",
        type=str,
        help="3R Scan configuration file name",
    )
    parser.add_argument(
        "--split",
        dest="split",
        default="train",
        type=str,
        help="split to run subscan generation on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp/sparse/0",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
    )
    parser.add_argument("--vis_dir", type=str, default="vis")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=20)
    return parser.parse_known_args()


if __name__ == "__main__":
    common.init_log()
    _LOGGER.info("**** Starting converting 3RScan to COLMAP format ****")

    args, unknown = parse_args()
    split = args.split
    os.makedirs(args.vis_dir, exist_ok=True)

    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    root_dir = cfg.data.root_dir
    output_dir = args.output_dir

    scan_ids = process_data(cfg, split=split)
