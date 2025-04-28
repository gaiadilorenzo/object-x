import functools
import os
import os.path as osp
import shutil
from argparse import Namespace
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


class ImageConfig(BaseModel):
    img_step: int = 5
    w: int = 960
    h: int = 540


class PreprocessConfig(BaseModel):
    pc_resolutions: List[int] = Field(default_factory=lambda: [64, 128, 256, 512])
    subscenes_per_scene: int = 7
    filter_segment_size: int = 512
    min_obj_points: int = 50
    anchor_type_name: str = ""


class ImageEncodingConfig(BaseModel):
    resize_w: int = 1024
    resize_h: int = 576
    img_rotate: bool = True
    patch_w: int = 16
    patch_h: int = 9
    record_feature: bool = False
    use_feature: bool = False
    preload_feature: bool = False
    feature_dir: str = ""


class CrossSceneConfig(BaseModel):
    use_cross_scene: bool = False
    num_scenes: int = 0
    num_negative_samples: int = 0
    use_tf_idf: bool = False


class SceneGraphConfig(BaseModel):
    obj_img_patch: str = ""
    obj_patch_num: int = 1
    obj_topk: int = 10
    use_predicted: bool = False


class AuxiliaryConfig(BaseModel):
    use_patch_depth: bool = False
    depth_dir: str = ""


class ImageAugmentationConfig(BaseModel):
    rotation: float = 60.0
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    color: float = 0.3


class PCSAugmentationConfig(BaseModel):
    granularity: List[float] = Field(default_factory=lambda: [0.05, 0.2, 0.4])
    magnitude: List[float] = Field(default_factory=lambda: [0.2, 0.4, 0.4])


class DataAugmentationConfig(BaseModel):
    use_aug: bool = False
    img: ImageAugmentationConfig = Field(default_factory=ImageAugmentationConfig)
    use_aug_3D: bool = False
    pcs: PCSAugmentationConfig = Field(default_factory=PCSAugmentationConfig)


class MetricsConfig(BaseModel):
    all_k: List[int] = [1, 2, 3, 4, 5]


class DataConfig(BaseModel):
    name: Literal["Scan3R", "ScanNet"] = "Scan3R"
    root_dir: str = Field(default_factory=lambda: os.getenv("DATA_ROOT_DIR", ""))
    rescan: bool = False
    temporal: bool = False
    resplit: bool = False
    from_gt: bool = True
    debug_few_scans: Optional[int] = None
    preload_masks: bool = False
    preload_slat: bool = True
    img: ImageConfig = Field(default_factory=ImageConfig)
    cross_scene: CrossSceneConfig = Field(default_factory=CrossSceneConfig)
    scene_graph: SceneGraphConfig = Field(default_factory=SceneGraphConfig)
    auxiliary: AuxiliaryConfig = Field(default_factory=AuxiliaryConfig)
    img_encoding: ImageEncodingConfig = Field(default_factory=ImageEncodingConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)


# Define model classes
class BackboneConfig(BaseModel):
    name: str = "GCViT"
    cfg_file: str = ""
    pretrained: str = ""
    use_pretrained: bool = True
    num_reduce: int = 1
    backbone_dim: int = 512


class PatchConfig(BaseModel):
    hidden_dims: List[int] = Field(default_factory=list)
    encoder_dim: int = 256
    gcn_layers: int = 0


class ObjConfig(BaseModel):
    embedding_dim: int = 256
    embedding_hidden_dims: List[int] = Field(default_factory=list)
    encoder_dim: int = 256


class OtherConfig(BaseModel):
    drop: float = 0.0


class VoxelConfig(BaseModel):
    in_feature_dim: int = 8
    out_feature_dim: int = 8
    in_channel_res: int = 64
    out_channel_res: int = 32
    pretrained: str = "pretrained/slat_pretrained.pth.tar"
    channels: list[int] = [16]


class EncoderConfig(BaseModel):
    net: str = "sgaligner"
    freeze: bool = False

    modules: List[str] = Field(default_factory=list)
    rel_dim: int = 9
    attr_dim: int = 164
    img_emb_dim: int = 256
    img_patch_feat_dim: int = 768
    multi_view_aggregator: Optional[str] = None
    use_pos_enc: bool = False
    label_file_name: str = ""
    pred_subfix: str = "inseg.ply"
    use_predicted: bool = False
    registration: bool = False
    scan_type: str = "scan"
    img_transformer: bool = False
    global_descriptor_dim: int = 1024
    alignment_threshold: float = 0.4

    use_pretrained: bool = True
    pretrained: str = "pretrained/u3dgs_pretrained_16.pth.tar"
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    obj: ObjConfig = Field(default_factory=ObjConfig)
    other: OtherConfig = Field(default_factory=OtherConfig)
    voxel: VoxelConfig = Field(default_factory=VoxelConfig)


class DecoderConfig(BaseModel):
    net: str = "sparsestructure"
    modules: List[str] = Field(default_factory=list)


class AutoencoderConfig(BaseModel):
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)


# Define training classes
class OptimConfig(BaseModel):
    lr: float = 0.001
    scheduler: str = "step"
    lr_decay: float = 0.97
    lr_decay_steps: int = 1
    lr_min: float = 0.0005
    T_max: int = 1000
    T_mult: int = 1
    weight_decay: float = 0.000001
    max_epoch: int = 10500
    free_backbone_epoch: int = 10500
    grad_acc_steps: int = 1
    sched_start_epoch: int = 150_000
    sched_end_epoch: int = 300_000
    end_lr: float = 0.0001
    max_grad_norm: float = 10.0

    # Scene Graph Localizer
    free_sgaligner_epoch: int = -1
    free_voxel_epoch: int = -1


class LossConfig(BaseModel):
    use_temporal: bool = False
    loss_type: str = "ICLLoss"
    alpha: float = 0.5
    temperature: float = 0.1
    margin: float = 0.5
    epsilon: float = 1e-8
    alignment_loss_weight: float = 1.0
    constrastive_loss_weight: float = 1.0
    zoom: float = 0.1
    use_global_descriptor: bool = False
    global_loss_coef: float = 0.5
    global_desc_temp: float = 1.0
    decoder_weight: float = 1.0


class TrainConfig(BaseModel):
    rot_factor: float = 1.0
    augmentation_noise: float = 0.005
    clip_grad: Optional[float] = None
    gpus: int = 1
    precision: int = 16
    batch_size: int = 1
    num_workers: int = 1
    freeze_backbone: bool = False
    log_steps: int = 1
    snapshot_steps: int = 100
    optim: OptimConfig = Field(default_factory=OptimConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    use_vis: bool = False
    vis_epoch_steps: int = 100
    aux_channels: int = 0
    pc_res: int = 512
    max_grad_norm: float = 10.0
    train_decoder: bool = True
    freeze_encoder: bool = False
    data_aug: DataAugmentationConfig = Field(default_factory=DataAugmentationConfig)
    overfit: bool = False
    overfit_obj_ids: List[int] = Field(default_factory=list)
    visualize_steps: int = 1
    val_steps: int = 1
    inner_val_steps: int = 2000


# Scene Graph Localizer
class RoomRetrievalConfig(BaseModel):
    epsilon_th: float = 0.8
    method_name: str = ""


class ValConfig(BaseModel):
    batch_size: int = 1
    num_workers: int = 1
    pretrained: str = ""
    epsilon_th: float = 0.8
    method_name: str = ""
    visualize_steps: int = 50
    pc_res: int = 512
    overfit: bool = False
    data_mode: str = "orig"
    # Scene Graph Alignment
    overlap_low: float = 0.0
    overlap_high: float = 0.0
    # Scene Graph Localizer
    room_retrieval: RoomRetrievalConfig = Field(default_factory=RoomRetrievalConfig)
    overfit_obj_ids: List[int] = Field(default_factory=list)


class InferenceConfig(BaseModel):
    slat_model_path: str = "pretrained/slat_pretrained.pth.tar"
    ulat_model_path: str = "pretrained/u3dgs_pretrained_16_ot.pth.tar"
    output_dir: str = "files/gs_embeddings"


# Global configuration
class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    val: ValConfig = Field(default_factory=ValConfig)

    task: str = "reconstruction"

    # Scene Graph Localizer
    use_resume: bool = False
    resume: str = ""
    mode: str = "train"
    output_dir: str = ""
    seed: int = 42
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    # Scene Graph Alignment
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    registration: bool = False

    @property
    def snapshot_dir(self) -> str:
        return osp.join(self.output_dir, "snapshots")

    @property
    def log_dir(self) -> str:
        return osp.join(self.output_dir, "logs")

    @property
    def event_dir(self) -> str:
        return osp.join(self.output_dir, "events")


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


# Load configuration from YAML
def update_configs(
    file_path: str, args: Namespace, do_ensure_dir: bool = True
) -> Config:

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    config = Config(**data)
    for arg in args:
        key, value = arg.split("=")
        value = eval(value)
        rsetattr(config, key, value)

    if do_ensure_dir:
        ensure_dir(config.log_dir)
        ensure_dir(config.snapshot_dir)
        ensure_dir(config.event_dir)
        shutil.copyfile(file_path, osp.join(config.output_dir, "config.yaml"))

    return config
