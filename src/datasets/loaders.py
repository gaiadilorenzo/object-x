import random
from typing import Any

import numpy
import torch
from torch.utils import data

from configs import Config
from utils import torch_util


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_val_data_loader(
    cfg: Config, dataset: Any
) -> tuple[data.DataLoader, data.DataLoader]:
    _, train_dataloader = get_train_dataloader(cfg, dataset=dataset)
    _, val_dataloader = get_val_dataloader(cfg, dataset=dataset)
    return train_dataloader, val_dataloader


def get_train_dataloader(
    cfg: Config, dataset: Any
) -> tuple[data.Dataset, data.DataLoader]:
    train_dataset = dataset(cfg, split="train")
    train_dataloader = torch_util.build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return train_dataset, train_dataloader


def get_val_dataloader(
    cfg: Config, dataset: Any
) -> tuple[data.Dataset, data.DataLoader]:
    val_dataset = dataset(cfg, split="val")
    val_dataloader = torch_util.build_dataloader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return val_dataset, val_dataloader


def get_test_dataloader(
    cfg: Config, dataset: Any
) -> tuple[data.Dataset, data.DataLoader]:
    test_dataset = dataset(cfg, split="test")
    test_dataloader = torch_util.build_dataloader(
        test_dataset,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return test_dataset, test_dataloader
