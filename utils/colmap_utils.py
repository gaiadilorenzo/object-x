import os
import os.path as osp
from typing import Dict, List

import numpy as np
import open3d as o3d

from utils.geometry import rotation_matrix_to_quaternion


def write_points3D_txt(
    output_dir: str,
    obj_file: str,
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
    pointcloud_dir = os.path.join(output_dir, "sparse", "0")

    os.makedirs(pointcloud_dir, exist_ok=True)
    with open(os.path.join(pointcloud_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (point, color) in enumerate(zip(points, colors)):
            x, y, z = point
            r, g, b = (color * 255).astype(int)
            f.write(f"{i+1} {x} {y} {z} {r} {g} {b} 0.0\n")


def write_cameras_txt(output_dir: str, intrinsics: Dict[str, np.ndarray]) -> None:
    """
    Write camera intrinsics to COLMAP format.

    Args:
        intrinsics (Dict[str, np.ndarray]): Dictionary with keys: fx, fy, cx, cy (focal lengths and principal point).
        scan_id (str): Scan identifier.
    """

    cameras_dir = os.path.join(output_dir, "sparse", "0")
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
    output_dir: str,
    frame_poses: Dict[str, np.ndarray],
    frame_paths: List[str],
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

    images_dir = os.path.join(output_dir, "sparse", "0")
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
            full_path = path
            f.write(f"{idx+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {full_path}\n")
            f.write("0.0 0.0 -1\n")
