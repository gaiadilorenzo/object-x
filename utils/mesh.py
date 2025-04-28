import json
import os

import numpy as np
import open3d as o3d
import psutil
import torch
from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.strategy import DefaultStrategy
from plyfile import PlyData, PlyElement

from utils.geometry import pose_quatmat_to_rotmat


def load_ply_to_pt(ply_path, output_pt_path):
    """
    Loads a PLY file and converts it to a .pt file containing a dictionary
    with the Gaussian parameters.  This reverses the process of `save_ply`.

    Args:
        ply_path (str): Path to the input .ply file.
        output_pt_path (str): Path to save the output .pt file.
    """

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    # Extract data.  Handles different naming conventions.
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    try:
        sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1)
        num_sh_coeffs = 45  # Assume SH degree 3 (1 + 2*3 + 1)**2 = 16 coefficients. 3 channels, 48 total. We already have the first 3.
        try:
            f_rest = np.stack(
                [vertex[f"f_rest_{i}"] for i in range(num_sh_coeffs)], axis=-1
            )
        except ValueError:
            f_rest = None
    except KeyError:
        # Handle cases where SH coefficients are named differently (e.g., gs-viewer)
        try:
            sh0 = np.stack(
                [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
            )
            f_rest_names = [
                name for name in vertex.data.dtype.names if name.startswith("f_rest_")
            ]
            f_rest_names.sort()  # Ensure consistent order
            f_rest = np.stack([vertex[name] for name in f_rest_names], axis=-1)
        except KeyError:
            # Handle cases where SH coefficients are named differently (e.g., different rendering libraries, supergaussian)
            try:
                sh0 = np.stack(
                    [vertex["sh_0"], vertex["sh_1"], vertex["sh_2"]], axis=-1
                )
                sh_names = [
                    name
                    for name in vertex.data.dtype.names
                    if name.startswith("sh_") and name not in ["sh_0", "sh_1", "sh_2"]
                ]
                sh_names.sort()  # Ensure consistent order
                f_rest = np.stack([vertex[name] for name in sh_names], axis=-1)

            except KeyError:
                raise KeyError(
                    "Spherical harmonics coefficients not found in PLY file. Need f_dc_0, f_dc_1, f_dc_2, and f_rest_i or sh_0, sh_1, sh_2 and sh_i"
                )

    opacities = vertex["opacity"]
    scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
    )
    try:
        quats = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=-1,
        )
    except KeyError:
        # Handle cases where quats are named differently (e.g. supergaussian, which uses q_0, q_1, q_2, q_3 for rotation)
        try:
            quats = np.stack(
                [vertex["q_0"], vertex["q_1"], vertex["q_2"], vertex["q_3"]], axis=-1
            )
        except:
            raise KeyError(
                "Quaternions not found in PLY file. Need rot_0, rot_1, rot_2, rot_3 or q_0, q_1, q_2, q_3"
            )

    # Reshape SH coefficients
    sh0 = sh0.reshape(-1, 1, 3)  # [N, 1, 3]
    if f_rest is None:
        f_rest = np.zeros((sh0.shape[0], 15, 3), sh0.dtype)
    else:
        f_rest = f_rest.reshape(-1, 15, 3)  # [N, 15, 3], assuming SH degree 3
    sh = np.concatenate((sh0, f_rest), axis=1)  # [N, 16, 3]
    sh = torch.from_numpy(sh).transpose(1, 2)  # [N, 3, 16]

    # Create a dictionary to store the data
    splats = {
        "means": torch.from_numpy(xyz).float(),
        "sh0": torch.from_numpy(sh0).float(),  # Keep sh0 to be able to save the model
        "shN": torch.from_numpy(
            f_rest
        ).float(),  # Keep shN to be able to save the model
        "opacities": torch.from_numpy(opacities).float(),
        "scales": torch.from_numpy(scales).float(),
        "quats": torch.from_numpy(quats).float(),
        "sh": sh,
    }

    container = {"splats": splats}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)

    # Save the dictionary to a .pt file
    torch.save(container, output_pt_path)
    print(f"Converted {ply_path} to {output_pt_path}")


def construct_list_of_attributes(splats: dict):
    """
    Constructs the list of attributes for the PLY file, to be able to use with `save_ply`.
    """
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(splats["sh0"].shape[1]):
        attributes.append("f_dc_{}".format(i))
    for i in range(splats["shN"].shape[1]):
        attributes.append("f_rest_{}".format(i))
    attributes.append("opacity")
    for i in range(splats["scales"].shape[1]):
        attributes.append("scale_{}".format(i))
    for i in range(splats["quats"].shape[1]):
        attributes.append("rot_{}".format(i))
    return attributes


def save_ply(splats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (
        splats["sh0"]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        splats["shN"]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = splats["opacities"].detach().unsqueeze(-1).cpu().numpy()
    scale = splats["scales"].detach().cpu().numpy()
    rotation = splats["quats"].detach().cpu().numpy()

    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(splats)
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def process_sample(sample):
    """Convert a sample dictionary to the required JSON format."""

    # Extract intrinsic parameters
    fx = sample["K"][0, 0, 0].item()  # Focal length in x
    fy = sample["K"][0, 1, 1].item()  # Focal length in y

    # Extract image properties
    img_name = sample["image_name"][0].split(".")[0]  # Image name
    img_id = int(sample["image_id"].item())  # Convert tensor to int
    height, width = sample["image"].shape[-3:-1]  # Get image dimensions

    # Extract camera position (translation)
    camtoworld = sample["camtoworld"][0].numpy()  # Convert tensor to NumPy array
    position = camtoworld[:3, 3].tolist()  # Extract translation vector

    # Extract rotation (3x3 upper-left matrix)
    rotation = camtoworld[:3, :3].tolist()  # Convert rotation to list

    # Construct JSON entry
    json_entry = {
        "id": img_id,
        "img_name": img_name,
        "width": width,
        "height": height,
        "position": position,
        "rotation": rotation,
        "fx": fx,
        "fy": fy,
    }

    return json_entry


def rasterize_splats(
    splats,
    world_to_cams,
    Ks,
    width,
    height,
    strategy,
    masks=None,
    type_="3dgs",
    **kwargs,
):

    means = splats["means"]  # [N, 3]
    # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
    # rasterization does normalization internally
    quats = splats["quats"] if "quats" in splats else None  # [N, 4]
    scales = torch.exp(splats["scales"]) if "scales" in splats else None  # [N, 3]
    covars = splats["covars"] if "covars" in splats else None  # [N, 3, 3]
    if scales is not None and scales.shape[-1] == 2:
        # concat a zero vector
        scales = torch.cat([scales, torch.zeros_like(scales[:, :1])], -1)
    opacities = torch.sigmoid(splats["opacities"])  # [N,]

    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

    rasterize_mode = "classic"
    if type_ == "3dgs":
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=world_to_cams,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            covars=covars,
            absgrad=(
                strategy.absgrad if isinstance(strategy, DefaultStrategy) else False
            ),
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model="pinhole",
            **kwargs,
        )
    else:
        render_colors, render_alphas, info = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=world_to_cams,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=(
                strategy.absgrad if isinstance(strategy, DefaultStrategy) else False
            ),
            sparse_grad=False,
            **kwargs,
        )

    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info


def create_meshes_from_depth_maps(
    color_images,
    depth_maps,
    intrinsics,
    camtoworlds,
    method="tsdf",
    ball_radius=0.05,
    alpha=0.1,
    voxel_size=0.05,
    tsdf_volume=None,
):
    """
    Creates a single, fused mesh from multiple depth maps.

    Args:
        depth_maps: (N, H, W) NumPy array of depth maps (float32).
        intrinsics: (N, 3, 3) or (3, 3) NumPy array/Tensor of intrinsic matrices.
        camtoworlds: (N, 4, 4) NumPy array/Tensor of camera-to-world matrices.
        depth_threshold: Threshold for depth discontinuity handling.

    Returns:
        An Open3D TriangleMesh object representing the fused mesh.
    """

    # --- Ensure NumPy Arrays ---
    if isinstance(depth_maps, torch.Tensor):
        depth_maps = depth_maps.cpu().numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(camtoworlds, torch.Tensor):
        camtoworlds = camtoworlds.cpu().numpy()

    num_images = depth_maps.shape[0]
    height = depth_maps.shape[1]
    width = depth_maps.shape[2]

    # --- Handle Intrinsic Matrix ---
    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics, (num_images, 1, 1))
    elif intrinsics.ndim != 3 or intrinsics.shape[0] != num_images:
        raise ValueError("Intrinsics must be (N, 3, 3) or (3, 3)")

    # --- Handle camtoworlds ---
    if (
        camtoworlds.ndim != 3
        or camtoworlds.shape[0] != num_images
        or camtoworlds.shape[1] != 4
        or camtoworlds.shape[2] != 4
    ):
        raise ValueError("camtoworlds must be a (N, 4, 4) array")

    # --- Extract Intrinsic Parameters (Ensure NumPy) ---
    fx = intrinsics[:, 0, 0]  # (N,)
    fy = intrinsics[:, 1, 1]  # (N,)
    cx = intrinsics[:, 0, 2]  # (N,)
    cy = intrinsics[:, 1, 2]  # (N,)

    # --- Create Meshgrid ---
    uu, vv = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uu = np.tile(uu.flatten(), (num_images, 1))  # (N, H*W)
    vv = np.tile(vv.flatten(), (num_images, 1))  # (N, H*W)

    # --- Back-projection ---
    depth_values = depth_maps.reshape(num_images, -1)  # (N, H*W)
    valid_depth_mask = depth_values > 0  # (N, H*W)

    x_n = (uu - cx[:, None]) / fx[:, None]  # (N, H*W)
    y_n = (vv - cy[:, None]) / fy[:, None]  # (N, H*W)
    X = x_n * depth_values  # (N, H*W)
    Y = y_n * depth_values  # (N, H*W)
    Z = depth_values  # (N, H*W)

    # --- Camera Coordinates (Homogeneous) ---
    points_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)  # (N, H*W, 4)

    # --- Transform to World Coordinates ---
    points_cam = points_cam.reshape(num_images, -1, 4, 1)  # (N, H*W, 4, 1)
    points_world = camtoworlds[:, None, :, :] @ points_cam  # (N, H*W, 4, 1)
    points_world = points_world.reshape(num_images, -1, 4)  # (N, H*W, 4)
    points_world = points_world[..., :3] / points_world[..., 3:]  # (N, H*W, 3)

    # --- Combine Point Clouds ---
    all_points_world = points_world.reshape(-1, 3)  # (N*H*W, 3)
    all_valid_mask = valid_depth_mask.flatten()  # (N*H*W,)
    all_points_world = all_points_world[all_valid_mask]  # remove invalid points

    # --- Create Open3D PointCloud ---
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_points_world)

    # --- Triangulation (Choose based on 'method') ---
    if method == "ball_pivoting":
        # --- Voxel Downsampling ---
        downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size=0.01)
        # --- Normal Estimation ---
        downsampled_pcd.estimate_normals()

        # Ball Pivoting Algorithm
        if isinstance(ball_radius, (int, float)):
            radii = [ball_radius]
        else:
            radii = ball_radius
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                downsampled_pcd, o3d.utility.DoubleVector(radii)
            )
    elif method == "alpha_shapes":
        # --- Voxel Downsampling ---
        downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
        # --- Normal Estimation ---
        downsampled_pcd.estimate_normals()

        # Alpha Shapes
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                downsampled_pcd, alpha
            )
    elif method == "poisson":
        # --- Voxel Downsampling ---
        downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
        # --- Normal Estimation ---
        downsampled_pcd.estimate_normals()

        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                downsampled_pcd, depth=14
            )
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    elif method == "marching_cubes":
        # Marching Cubes directly on the point cloud (requires voxelization)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_marching_cubes(
            combined_pcd, voxel_size=voxel_size
        )
    elif method == "tsdf":
        # TSDF Integration and Marching Cubes
        if tsdf_volume is None:
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size,
                sdf_trunc=0.04,  # Adjust truncation distance as needed
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                if color_images is not None
                else o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            )

        # Integrate each depth map into the TSDF
        num_images = depth_maps.shape[0]
        for i in range(num_images):
            # Need a color image for integration, even if we don't use colors.
            image = (color_images[i] * 255).astype(np.uint8)
            color_image = o3d.geometry.Image(image)  # Need the color in image format
            depth_image = o3d.geometry.Image(
                depth_maps[i]
            )  # Need the depth in image format

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1.0,  # IMPORTANT: Ensure this matches your depth units!
                depth_trunc=5.0,  # Cull values greater than depth_trunc meters,
                convert_rgb_to_intensity=False,
            )

            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                width, height, fx[i], fy[i], cx[i], cy[i]
            )

            if tsdf_volume is not None:
                tsdf_volume.integrate(
                    rgbd, intrinsic_o3d, np.linalg.inv(camtoworlds[i])
                )
            else:
                volume.integrate(rgbd, intrinsic_o3d, np.linalg.inv(camtoworlds[i]))

        mesh = None
        if tsdf_volume is None:
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
    else:
        raise ValueError(
            "Invalid 'method'.  Must be 'ball_pivoting' or 'alpha_shapes'."
        )

    return mesh


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


def remove_small_components(mesh, min_triangles=100):
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


def splat_to_mesh(
    splat,
    Ks,
    world_to_cams,
    width,
    height,
    sh_degree_to_use,
    near_plane,
    far_plane,
    masks=None,
    voxel_size=0.015,
    sdf_trunc=0.04,
    type_="3dgs",
    pred_colors=None,
    pred_depths=None,
    device="cuda",
):
    """Converts a single splat to a mesh using the given parameters."""

    strategy = DefaultStrategy(verbose=True)
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,  # Adjust truncation distance as needed
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    cam_to_worlds = (
        torch.from_numpy(np.linalg.inv(pose_quatmat_to_rotmat(world_to_cams)))
        .to(device)
        .float()
    )  # Shape [C, 4, 4]

    R = torch.linalg.inv(cam_to_worlds[..., :3, :3].transpose(-2, -1))
    T = cam_to_worlds[..., :3, 3]
    world_to_cams = torch.cat([R, T.unsqueeze(-1)], dim=-1)
    world_to_cams = torch.cat(
        [
            world_to_cams,
            torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=cam_to_worlds.device)
            .view(1, 1, 4)
            .repeat(world_to_cams.shape[0], 1, 1),
        ],
        dim=1,
    )
    Ks = (
        torch.from_numpy(Ks)
        .unsqueeze(0)
        .repeat(world_to_cams.shape[0], 1, 1)
        .to(device)
        .float()
    )  # Shape [C, 3, 3]

    device = cam_to_worlds.device
    for i in range(len(world_to_cams)):
        K = Ks[i : i + 1]
        world_to_cam = world_to_cams[i : i + 1].to(device)
        sh_degree_to_use = sh_degree_to_use

        # json_entry = process_sample(sample)
        # json_data.append(json_entry)

        # forward
        if pred_colors is None and pred_depths is None:
            renders, alphas, info = rasterize_splats(
                splat,
                world_to_cams=world_to_cam,
                Ks=K,
                width=int(width),
                height=int(height),
                sh_degree=sh_degree_to_use,
                near_plane=near_plane,
                far_plane=far_plane,
                render_mode="RGB+ED",
                strategy=strategy,
                masks=None,
                type_=type_,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
        else:
            colors, depths = (
                pred_colors[i : i + 1].permute(0, 2, 3, 1),
                pred_depths[i : i + 1],
            )

        create_meshes_from_depth_maps(
            np.ascontiguousarray(colors.cpu().numpy()),
            depths.cpu(),
            K.cpu(),
            torch.linalg.inv(world_to_cam).cpu(),
            tsdf_volume=tsdf_volume,
        )
        del colors, depths

    print("Creating Mesh...")
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print("Refining Mesh...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    print("Calculate Normals...")
    mesh.compute_vertex_normals()
    # --- Save the Mesh ---
    return mesh
