import json
import os
from typing import Any, List

import torch
from safetensors.torch import load_file
from torch import nn

from configs import AutoencoderConfig
from src.models.backbones import (
    SLatEncoder,
    SLatGaussianDecoder,
    SparseStructureDecoder,
    SparseStructureEncoder,
)
from src.modules.sparse.basic import SparseTensor
from src.representations.gaussian.gaussian_model import Gaussian

SCRATCH = os.environ.get("SCRATCH", "/scratch")


class LatentAutoencoder(nn.Module):
    def __init__(
        self,
        cfg: AutoencoderConfig,
        device: str = "cuda",
        downsample: bool = False,
        load_pretrained: bool = True,
    ) -> None:
        super(LatentAutoencoder, self).__init__()
        self.cfg = cfg
        json_path = f"{SCRATCH}/TRELLIS-image-large/pipeline.json"
        with open(json_path, "r") as f:
            trellis_pipeline = json.load(f)

        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["slat_encoder"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            self.encoder = SLatEncoder(**configs["args"])
            self.encoder.load_state_dict(state_dict, strict=False)
            self.encoder = self.encoder.to(device)
        else:
            self.encoder = SLatEncoder(
                resolution=64,
                in_channels=768,
                model_channels=768,
                latent_channels=8,
                num_blocks=4,
                num_heads=4,
                use_fp16=True,
            ).to(device)

        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["slat_decoder_gs"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            net = SLatGaussianDecoder(**configs["args"])
            net.load_state_dict(state_dict)

        else:
            net = SLatGaussianDecoder(
                resolution=64,
                model_channels=768,
                latent_channels=8,
                num_blocks=4,
                num_heads=4,
                num_head_channels=64,
                use_fp16=True,
            ).to(device)

        self.decoder = net.to(device)

    def encode(self, data_dict: dict[str, Any]) -> SparseTensor:
        data_dict = data_dict["scene_graphs"]
        voxel_sparse_tensor = data_dict["tot_obj_splat"]
        return self.encoder(voxel_sparse_tensor)

    def decode(self, code: SparseTensor) -> List[Gaussian]:
        return self.decoder(code)
