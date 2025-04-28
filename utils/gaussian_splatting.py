from typing import List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement

from utils.colmap import rotmat2qvec
from utils.general_utils import build_scaling_rotation, inverse_sigmoid, strip_symmetric


class GaussianSplat:
    max_sh_degree = 3

    def __init__(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        features_rest: Optional[torch.Tensor] = None,
    ):
        self.xyz = xyz.reshape(-1, 3)
        self.features_dc = features_dc.reshape(-1, 1, 3)
        if features_rest is None:
            features_rest = torch.zeros(
                (self.xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1),
                device=xyz.device,
            )
        self.features_rest = features_rest.reshape(
            -1, 3, (self.max_sh_degree + 1) ** 2 - 1
        )
        self.opacity = opacity.reshape(-1, 1)
        self.scaling = scaling.reshape(-1, 3)
        self.rotation = rotation.reshape(-1, 4)
        self.max_sh_degree = 3
        self.active_sh_degree = 0

        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self.scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self.rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    @property
    def get_features(
        self,
    ) -> torch.Tensor:  # Shape: (batch_size, 3 + 3 * ((max_sh_degree + 1) ** 2 - 1))
        features_dc = self.features_dc.reshape(-1, 1, 3)
        features_rest = self.features_rest.reshape(
            -1, (self.max_sh_degree + 1) ** 2 - 1, 3
        )
        return features_dc

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self.opacity).reshape(-1, 1)

    @property
    def dtype(self) -> List[Tuple[str, str]]:
        return [(attribute, "f4") for attribute in self._construct_list_of_attributes()]

    def _construct_list_of_attributes(self) -> List[str]:
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self.features_dc.shape[-2] * self.features_dc.shape[-1]):
            l.append("f_dc_{}".format(i))
        for i in range(self.features_rest.shape[-2] * self.features_rest.shape[-1]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[-1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[-1]):
            l.append("rot_{}".format(i))
        return l

    @classmethod
    def load_ply(cls, path: str) -> "GaussianSplat":
        """Load gaussian splat from a ply file."""

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (cls.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return cls(
            xyz=xyz,
            features_dc=features_dc,
            opacity=opacities,
            scaling=scales,
            rotation=rots,
            features_rest=features_extra,
        )

    def save_ply(self, path: str, indices: Optional[torch.Tensor] = None) -> None:
        """Save gaussian splat to a ply file."""

        xyz = (self.xyz[indices] if indices is not None else self.xyz).reshape((-1, 3))
        # it torch tensor make sure to convert it to numpy
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self.features_dc[indices] if indices is not None else self.features_dc
        ).reshape(-1, 3)
        if isinstance(f_dc, torch.Tensor):
            f_dc = f_dc.detach().cpu().numpy()
        f_rest = (
            self.features_rest[indices] if indices is not None else self.features_rest
        ).reshape((-1, 3 * ((self.max_sh_degree + 1) ** 2 - 1)))
        if isinstance(f_rest, torch.Tensor):
            f_rest = f_rest.detach().cpu().numpy()
        opacities = (
            self.opacity[indices] if indices is not None else self.opacity
        ).reshape((-1, 1))
        if isinstance(opacities, torch.Tensor):
            opacities = opacities.detach().cpu().numpy()
        scale = (
            self.scaling[indices] if indices is not None else self.scaling
        ).reshape((-1, 3))
        if isinstance(scale, torch.Tensor):
            scale = scale.detach().cpu().numpy()
        rotation = (
            self.rotation[indices] if indices is not None else self.rotation
        ).reshape((-1, 4))
        if isinstance(rotation, torch.Tensor):
            rotation = rotation.detach().cpu().numpy()

        elements = np.empty(xyz.shape[0], dtype=self.dtype)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def to_tensor(self):
        num_points = self.xyz.shape[0]
        return np.concatenate(
            [
                self.xyz.reshape(num_points, -1),
                self.rotation.reshape(num_points, -1),
                self.scaling.reshape(num_points, -1),
                self.opacity.reshape(num_points, -1),
                self.features_dc.reshape(num_points, -1),
            ],
            axis=1,
        )

    @staticmethod
    def rx(theta):
        return np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def ry(theta):
        return np.matrix(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    @staticmethod
    def rz(theta):
        return np.matrix(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def rescale(self, scale: float):
        if scale != 1.0:
            self.xyz *= scale
            self.scaling += torch.log(scale)

    def get_scale(self):
        # return the scaling of the gaussian splat
        mean = torch.mean(self.xyz, dim=0)
        return torch.max(torch.abs(self.xyz - mean))

    def get_translation(self):
        return torch.mean(self.xyz, dim=0)

    def rotate_by_euler_angles(self, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0.0 and y == 0.0 and z == 0.0:
            return

        rotation_matrix = np.asarray(
            self.rx(x) @ self.ry(y) @ self.rz(z), dtype=np.float32
        )

        return self.rotate_by_matrix(rotation_matrix)

    def rotate_by_matrix(self, rotation_matrix, keep_sh_degree: bool = True):
        # rotate xyz
        self.xyz = torch.matmul(self.xyz, torch.from_numpy(rotation_matrix.T))

        # rotate gaussian
        # rotate via quaternions
        def quat_multiply(quaternion0, quaternion1):
            w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
            w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
            return torch.concatenate(
                (
                    -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                ),
                axis=-1,
            )

        quaternions = torch.from_numpy(rotmat2qvec(rotation_matrix)[np.newaxis, ...])
        rotations_from_quats = quat_multiply(self.rotation, quaternions)
        self.rotation = rotations_from_quats / torch.norm(
            rotations_from_quats, dim=-1, keepdim=True
        )

        # rotate via rotation matrix
        # gaussian_rotation = build_rotation(torch.from_numpy(self.rotations)).cpu()
        # gaussian_rotation = torch.from_numpy(rotation_matrix) @ gaussian_rotation
        # xyzw_quaternions = R.from_matrix(gaussian_rotation.numpy()).as_quat(canonical=False)
        # wxyz_quaternions = xyzw_quaternions
        # wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
        # rotations_from_matrix = wxyz_quaternions
        # self.rotations = rotations_from_matrix

        # TODO: rotate shs
        if keep_sh_degree is False:
            print("set sh_degree=0 when rotation transform enabled")
            self.sh_degrees = 0
        return torch.tensor(self.to_tensor()), torch.tensor(rotation_matrix)

    def translation(self, x: float, y: float, z: float):
        if x == 0.0 and y == 0.0 and z == 0.0:
            return self

        self.xyz += torch.tensor([x, y, z], device=self.xyz.device)
        return self

    def to_torch(self, device: str = "cuda"):
        self.xyz = torch.tensor(self.xyz).float().to(device)
        self.rotation = torch.tensor(self.rotation).to(device).float()
        self.scaling = torch.tensor(self.scaling).to(device).float()
        self.opacity = torch.tensor(self.opacity).to(device).float()
        self.features_dc = torch.tensor(self.features_dc).to(device).float()
        self.features_rest = torch.tensor(self.features_rest).to(device).float()
        return self

    def to_pt(self):

        xyz = self.xyz
        sh0 = self.features_dc
        f_rest = self.features_rest
        opacities = self.opacity
        scales = self.scaling
        quats = self.rotation
        assert quats.shape[-1] == 4

        # Reshape SH coefficients
        sh0 = sh0.reshape(-1, 1, 3)  # [N, 1, 3]
        if f_rest is None:
            f_rest = torch.zeros((sh0.shape[0], 15, 3), sh0.dtype)
        else:
            f_rest = f_rest.reshape(-1, 15, 3)  # [N, 15, 3], assuming SH degree 3
        sh = torch.concatenate((sh0, f_rest), axis=1)  # [N, 16, 3]
        sh = sh.transpose(1, 2)  # [N, 3, 16]

        splats = {
            "means": xyz.float(),
            "sh0": sh0.float(),  # Keep sh0 to be able to save the model
            "shN": f_rest.float(),  # Keep shN to be able to save the model
            "opacities": opacities.float().squeeze(),
            "scales": scales.float(),
            "quats": quats.float(),
            "sh": sh,
        }
        return splats


class ViewPoint:
    @classmethod
    def visualize_point_cloud(
        self,
        point_cloud: np.array,
        extrinsic_mat: np.array = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        intrinsic_mat: np.array = np.array([500, 0, 320, 0, 500, 240, 0, 0, 1]),
        width: int = 640,
        height: int = 480,
    ):
        """Visualize point cloud."""

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="b", marker="o"
        )
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        # return fig as np.array
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
