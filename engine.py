import logging
import os
import random

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, filename: str = "prism_training.log") -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filename=log_path,
        filemode="a",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("").addHandler(console)


class PolyLRScheduler:
    def __init__(self, base_lr: float, num_epochs: int, power: float = 0.9) -> None:
        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.power = power

    def __call__(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        lr = float(
            round(
                self.base_lr
                * np.power(
                    1 - np.float32(epoch) / np.float32(self.num_epochs), self.power
                ),
                8,
            )
        )
        optimizer.param_groups[0]["lr"] = lr
        return lr


class _RepeatSampler:
    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def get_num_classes(dataname: str) -> int:
    if dataname in ["BRATS2023", "BRATS2021", "BRATS2020", "BRATS2019", "BRATS2018"]:
        return 4
    if dataname == "BRATS2015":
        return 5
    raise ValueError(f"Unsupported dataset name: {dataname}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def core_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
