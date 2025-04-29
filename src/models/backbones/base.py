# This code has been adapted from https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_vae/base.py

from typing import *

import torch
import torch.nn as nn

from src.modules import sparse as sp
from src.modules.sparse.transformer import SparseTransformerBlock
from src.modules.transformer import AbsolutePositionEmbedder
from src.modules.utils import convert_module_to_f16, convert_module_to_f32


def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (
                16 * (i % 2),
            ) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (
                0,
                0,
                0,
            ), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (
                i % 2
            ), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,  # So ideally this is my embedding size (8)
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    shift_sequence=shift_sequence,
                    shift_window=shift_window,
                    serialize_mode=serialize_mode,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    qk_rms_norm=self.qk_rms_norm,
                )
                for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(
                    self
                )
            ]
        )

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x)  # Maps channels to model_channels
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(
                x.coords[:, 1:]
            )  # Add absolute position embedding
        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h)  # Apply transformer block
        return h


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(
        self, init_type="normal", gain=0.02, bias_value=0.0, target_op=None
    ):
        """
        initialize network's weights
        init_type: normal | xavier_normal | kaiming | orthogonal | xavier_unifrom
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__

            if target_op is not None:
                if classname.find(target_op) == -1:
                    return False

            if hasattr(m, "param_inited"):
                return

            # print('classname',classname)
            if hasattr(
                m, "weight"
            ):  # and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "xavier_unifrom":
                    nn.init.xavier_uniform_(m.weight.data, gain=gain)
                elif init_type == "constant":
                    nn.init.constant_(m.weight.data, gain)
                else:
                    raise NotImplementedError()

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, bias_value)
            m.param_inited = True

        self.init_apply(init_func)

    def getParamList(self, x):
        return list(x.parameters())

    def init_apply(self, fn):
        for m in self.children():
            if hasattr(m, "param_inited"):
                if m.param_inited is False:
                    m.init_apply(fn)
            else:
                m.apply(fn)
        fn(self)
        return self


class mySequential(nn.Sequential, BaseNetwork):
    def __init__(self, *args):
        super(mySequential, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
