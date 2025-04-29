import random

import numpy as np
import scipy
import torch


class ElasticDistortion:
    """
    Apply elastic distortion on sparse coordinate space.
    First projects the position onto a voxel grid and then apply the distortion to the voxel grid.
    """

    def __init__(
        self,
        apply_distorsion: bool = True,
        granularity: list[float] = [0.2, 0.8],
        magnitude: list[float] = [0.4, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(
        coords: torch.Tensor, granularity: float, magnitude: float
    ) -> torch.Tensor:
        coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=0, fill_value=0
        )
        coords = coords + interp(coords) * magnitude
        return torch.tensor(coords).float()

    def __call__(self, pcs_pos: torch.Tensor) -> torch.Tensor:
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    pcs_pos = ElasticDistortion.elastic_distortion(
                        pcs_pos,
                        self._granularity[i],
                        self._magnitude[i],
                    )
        return pcs_pos

    def __repr__(self) -> str:
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__,
            self._apply_distorsion,
            self._granularity,
            self._magnitude,
        )
