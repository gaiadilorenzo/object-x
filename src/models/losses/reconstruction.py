import warnings
from math import exp

import lpips
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.autograd import Variable
from tqdm.auto import tqdm


def evaluate_mesh(
    pcd_ground_truth, mesh_estimated, thresholds=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
):
    # Completeness: Distance from ground truth points to mesh
    distances_gt_to_mesh = point_to_mesh_distances(pcd_ground_truth, mesh_estimated)

    # Accuracy: Distance from each mesh vertex to ground truth point cloud
    mesh_vertices_pcd = o3d.geometry.PointCloud()
    mesh_vertices_pcd.points = mesh_estimated.vertices  # Use mesh vertices directly
    distances_mesh_to_gt = point_to_point_distances(mesh_vertices_pcd, pcd_ground_truth)

    metrics = {}
    for threshold in thresholds:
        completeness = (
            (distances_gt_to_mesh < threshold).sum() / len(distances_gt_to_mesh) * 100
        )
        metrics[f"completeness_{threshold}"] = completeness

        accuracy = (
            (distances_mesh_to_gt < threshold).sum() / len(distances_mesh_to_gt) * 100
        )
        metrics[f"accuracy_{threshold}"] = accuracy

        if accuracy + completeness > 0:
            f1_score = 2 * (accuracy * completeness) / (accuracy + completeness)
        else:
            f1_score = 0.0
        metrics[f"f1_{threshold}"] = f1_score

    return metrics


def point_to_point_distances(
    pcd_source: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud
) -> np.ndarray:
    """
    Calculates the distances from each point in the source point cloud to the closest
    point in the target point cloud using Open3D's KDTree.

    Args:
        pcd_source: The source Open3D PointCloud.
        pcd_target: The target Open3D PointCloud.

    Returns:
        A NumPy array of distances.
    """

    # Convert target point cloud to KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd_target)

    # Compute nearest neighbor distances
    distances = []
    for point in np.asarray(pcd_source.points):
        _, idx, dist = kdtree.search_knn_vector_3d(point, 1)  # Find nearest neighbor
        distances.append(np.sqrt(dist[0]))  # Store distance

    return np.array(distances)


def point_to_mesh_distances(
    pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """
    Calculates distances from each point in a point cloud to the closest
    point on a mesh using Open3D's Tensor API.

    Args:
        pcd: The Open3D PointCloud.
        mesh: The Open3D TriangleMesh.

    Returns:
        A NumPy array of distances.
    """

    # Create a scene and add the mesh
    scene = o3d.t.geometry.RaycastingScene()
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_tensor)

    # Query points from the point cloud
    query_points = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)

    # Compute distances (unsigned)
    distances = scene.compute_distance(query_points)

    return distances.numpy()  # Convert to NumPy array


def mse_mask_loss(prediction, target, mask, weight=10.0):
    """
    Compute masked MSE loss, giving higher weight to important regions.

    Args:
        prediction (torch.Tensor): The predicted voxel grid.
        target (torch.Tensor): The ground truth voxel grid.
        mask (torch.Tensor): The binary mask indicating regions of interest (1 for features, 0 for background).
        weight (float): The weighting factor for non-zero regions.

    Returns:
        torch.Tensor: Computed loss value.
    """

    loss = (mask * (prediction - target) ** 2) + (1 - mask) * (
        prediction - target
    ) ** 2 / weight
    return loss.mean()


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class LPIPS(nn.Module):
    def __init__(self, device="cuda"):
        super(LPIPS, self).__init__()
        self.model = (
            lpips.LPIPS(net="alex").to(device)
            if torch.cuda.is_available()
            else lpips.LPIPS(net="alex")
        )

    def forward(self, gt, recon):
        # normalize between [-1, 1]
        gt = gt * 2 - 1
        recon = recon * 2 - 1
        return self.model(gt, recon).mean()
