import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from configs import AutoencoderConfig
from src.models.backbones import (
    CrossAttention3D,
    MultiModalEncoder,
    SparseStructureDecoder,
)
from src.modules.sparse.basic import SparseTensor
from src.modules.sparse.linear import SparseLinear

_LOGGER = logging.getLogger(__name__)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        cfg: AutoencoderConfig,
        device: str = "cuda",
        text_guidance: Any = None,
    ) -> None:
        super(AutoEncoder, self).__init__()
        self.cfg = cfg
        self.out_feature_dim = cfg.encoder.voxel.out_feature_dim
        self.channels = cfg.encoder.voxel.channels

        self.encoder = MultiModalEncoder(
            modules=cfg.encoder.modules,
            rel_dim=cfg.encoder.rel_dim,
            attr_dim=cfg.encoder.attr_dim,
            img_emb_dim=cfg.encoder.img_emb_dim,
            img_feat_dim=cfg.encoder.img_patch_feat_dim,
            dropout=cfg.encoder.other.drop,
            img_aggregation_mode=getattr(cfg.encoder, "multi_view_aggregator", None),
            use_pos_enc=getattr(cfg.encoder, "use_pos_enc", False),
            in_latent_dim=cfg.encoder.voxel.in_feature_dim,
            out_latent_dim=cfg.encoder.voxel.out_feature_dim,
            voxel_encoder=cfg.decoder.net,
            channels=cfg.encoder.voxel.channels,
        ).to(device)

        out_latent_dim = cfg.encoder.voxel.out_feature_dim
        in_latent_dim = cfg.encoder.voxel.in_feature_dim
        final_res = cfg.encoder.voxel.in_channel_res // 2 ** (len(self.channels) - 1)
        modules = []
        if len(cfg.encoder.modules) > 1:
            self.cross_attn = CrossAttention3D(
                context_dim=300,
                voxel_channels=out_latent_dim,
            ).cuda()
            self.unflatten = nn.Unflatten(
                1, (out_latent_dim, final_res, final_res, final_res)
            )
        modules.append(
            nn.Unflatten(1, (out_latent_dim, final_res, final_res, final_res))
        )
        modules.append(
            SparseStructureDecoder(
                out_channels=in_latent_dim,
                latent_channels=out_latent_dim,
                num_res_blocks=1,
                channels=self.channels,
            )
        )
        self.voxel_decoder = nn.Sequential(*modules).to(device)

    def encode(self, x):
        embs = self.encoder(x)
        return embs["joint"]

    def decode(self, code) -> torch.Tensor:
        structured_latent = code
        if len(self.cfg.encoder.modules) > 1:
            voxel = self.unflatten(structured_latent[:, :-300])
            context = structured_latent[:, -300:]
            structured_latent = self.cross_attn(voxel, context).reshape(
                voxel.shape[0], -1
            )
        structured_latent = self.voxel_decoder(structured_latent)
        return structured_latent
