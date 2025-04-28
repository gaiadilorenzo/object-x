# Adapted from https://github.com/hesom/depth_to_mesh/blob/master/depth_to_mesh/__init__.py

import itertools
import logging
import math
import pathlib

import cv2
import numpy as np
import open3d as o3d
import torch
from skimage.io import imread
from tqdm import tqdm

DEFAULT_CAMERA = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=528.0, fy=528.0, cx=319.5, cy=239.5
)

logger = logging.getLogger(__name__)


def _pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int32)
    y = np.linspace(0, height - 1, height).astype(np.int32)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def depth_file_to_mesh(
    image,
    mask,
    cameraMatrix=DEFAULT_CAMERA,
    cameraExtrinsics=np.eye(4),
    minAngle=3.0,
    sun3d=False,
    depthScale=1000.0,
):
    """
    Converts a depth image file into a open3d TriangleMesh object

    :param image: path to the depth image file
    :param mask: mask to apply to the depth image
    :param cameraMatrix: numpy array of the intrinsic camera matrix
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :param sun3d: Specify if the depth file is in the special SUN3D format
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """

    if isinstance(image, str):
        depth_raw = cv2.imread(image, cv2.IMREAD_UNCHANGED).astype("uint16")
    else:
        depth_raw = image.astype("uint16")
    mask = cv2.resize(
        mask.astype("uint16"),
        (depth_raw.shape[1], depth_raw.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    depth_raw = np.where(mask == 0, 0, depth_raw)
    width = depth_raw.shape[1]
    height = depth_raw.shape[0]
    if sun3d:
        depth_raw = np.bitwise_or(depth_raw >> 3, depth_raw << 13)

    depth_raw = depth_raw.astype("float32")
    depth_raw /= depthScale

    logger.debug("Image dimensions:%s x %s", width, height)
    logger.debug("Camera Matrix:%s", cameraMatrix)

    if cameraMatrix is None:
        camera = DEFAULT_CAMERA
    else:
        camera = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=cameraMatrix[0, 0],
            fy=cameraMatrix[1, 1],
            cx=cameraMatrix[0, 2],
            cy=cameraMatrix[1, 2],
        )
    return depth_to_mesh(
        depth_raw.astype("float32"), mask, camera, cameraExtrinsics, minAngle
    )


def remove_small_components(mesh, min_triangles=200):
    # Convert triangle adjacency into a graph of connected components
    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)

    # Keep only triangles in large enough clusters
    kept_triangles = [
        i
        for i, c in enumerate(triangle_clusters)
        if cluster_n_triangles[c] >= min_triangles
    ]

    # Extract the vertex indices used by these triangles
    triangle_vertices = np.asarray(mesh.triangles)[
        kept_triangles
    ]  # Get corresponding vertex indices
    unique_vertex_indices = np.unique(triangle_vertices)  # Get unique vertex indices
    mesh_filtered = mesh.select_by_index(unique_vertex_indices)
    return mesh_filtered


def depth_to_mesh(
    depth, mask, camera=DEFAULT_CAMERA, extrinsics=np.eye(4), minAngle=3.0
):
    """
    Vectorized version of converting a depth image to a mesh, filtering out invalid triangles.

    :param depth: np.array of type float32 containing the depth image
    :param camera: open3d.camera.PinholeCameraIntrinsic
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """
    logger.info("Reprojecting points...")

    # Move data to GPU
    depth = torch.tensor(depth, device="cuda", dtype=torch.float32)
    mask = torch.tensor(mask, device="cuda", dtype=torch.float32)

    K = torch.tensor(camera.intrinsic_matrix, device="cuda", dtype=torch.float32)
    K_inv = torch.linalg.inv(K)

    # Reproject valid points
    pixel_coords = torch.tensor(
        _pixel_coord_np(depth.shape[1], depth.shape[0]),
        device="cuda",
        dtype=torch.float32,
    )  # [3, H*W]
    cam_coords = (K_inv @ pixel_coords) * depth.flatten()

    # Generate indices for triangles
    h, w = depth.shape
    i, j = torch.meshgrid(
        torch.arange(h - 1, device="cuda"),
        torch.arange(w - 1, device="cuda"),
        indexing="ij",
    )
    i, j = i.flatten(), j.flatten()

    # Form two triangles for each quad
    idx_t1 = torch.stack([i * w + j, (i + 1) * w + j, i * w + (j + 1)], dim=-1)
    idx_t2 = torch.stack(
        [i * w + (j + 1), (i + 1) * w + j, (i + 1) * w + (j + 1)], dim=-1
    )

    # Combine triangle indices
    indices = torch.cat([idx_t1, idx_t2], dim=0)

    # Extract vertices for each triangle
    verts = cam_coords[:, indices]

    # Calculate normals
    v1 = verts[:, :, 1] - verts[:, :, 0]
    v2 = verts[:, :, 2] - verts[:, :, 0]
    normals = torch.cross(v1, v2, dim=0)
    normal_lengths = torch.norm(normals, dim=0)

    # Calculate angles
    centers = verts.mean(dim=2)
    center_lengths = torch.norm(centers, dim=0)
    valid_normals = normal_lengths > 0
    valid_centers = center_lengths > 0

    # Filter invalid triangles (zero-length normals or centers)
    valid = valid_normals & valid_centers

    # Recalculate angles only for valid triangles
    normals = normals[:, valid] / normal_lengths[valid]
    centers = centers[:, valid] / center_lengths[valid]
    angles = torch.rad2deg(
        torch.arcsin(torch.abs(torch.einsum("ij,ij->j", normals, centers)))
    )

    # Further filter by angle
    angle_valid = angles > minAngle
    indices = indices[valid][angle_valid].to(torch.int32)

    homo_cam_coords = torch.cat(
        (cam_coords, torch.ones((1, cam_coords.shape[1]), device="cuda")), dim=0
    )
    cam_coords = (
        torch.tensor(extrinsics, device="cuda", dtype=torch.float32) @ homo_cam_coords
    )[:3, :]

    # Move data back to CPU for Open3D
    indices = indices.cpu().numpy()
    cam_coords = cam_coords.cpu().numpy()

    # Create Open3D mesh
    indices = o3d.utility.Vector3iVector(np.ascontiguousarray(indices))
    points = o3d.utility.Vector3dVector(np.ascontiguousarray(cam_coords.transpose()))
    mesh = o3d.geometry.TriangleMesh(points, indices)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    mesh = remove_small_components(mesh)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20_000)
    return mesh
