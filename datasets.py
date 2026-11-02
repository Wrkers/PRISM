import ast
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .data.transforms import (
        Compose,
        Identity,
        NumpyType,
        RandCrop3D,
        RandomFlip,
        RandomIntensityChange,
        RandomRotion,
    )
except ImportError:
    from data.transforms import (
        Compose,
        Identity,
        NumpyType,
        RandCrop3D,
        RandomFlip,
        RandomIntensityChange,
        RandomRotion,
    )


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLIT_ROOT = os.path.join(PROJECT_DIR, "datalist")

MASK_ARRAY = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
        [False, True, True, False],
        [False, True, False, True],
        [False, False, True, True],
        [True, True, True, False],
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
        [True, True, True, True],
    ]
)


def build_default_train_transforms(crop_size=(128, 128, 128)) -> str:
    h, w, z = crop_size
    return (
        f"Compose([RandCrop3D(({h},{w},{z})), RandomRotion(10), "
        "RandomIntensityChange((0.1,0.1)), RandomFlip(0), "
        "NumpyType((np.float32, np.int64)),])"
    )


def build_default_test_transforms() -> str:
    return "Compose([NumpyType((np.float32, np.int64)),])"


DEFAULT_TRAIN_TRANSFORMS = build_default_train_transforms()
DEFAULT_TEST_TRANSFORMS = build_default_test_transforms()


def resolve_split_path(filename: str, split_root: Optional[str] = None) -> str:
    if os.path.isabs(filename):
        return filename
    split_root = split_root or DEFAULT_SPLIT_ROOT
    return os.path.join(split_root, filename)


def get_modal_indices(modal: str) -> np.ndarray:
    if modal == "flair":
        return np.array([0])
    if modal == "t1ce":
        return np.array([1])
    if modal == "t1":
        return np.array([2])
    if modal == "t2":
        return np.array([3])
    return np.array([0, 1, 2, 3])


class BratsTrainNpyDataset(Dataset):
    def __init__(
        self,
        transforms: str = "",
        root: Optional[str] = None,
        split_root: Optional[str] = None,
        modal: str = "all",
        num_classes: int = 4,
        train_file: str = "train.txt",
    ) -> None:
        split_path = resolve_split_path(train_file, split_root)
        with open(split_path, "r", encoding="utf-8") as file:
            datalist = [line.strip() for line in file.readlines()]

        self.volpaths = [
            os.path.join(root, "vol", f"{case_name}_vol.npy") for case_name in datalist
        ]
        self.segpaths = [
            os.path.join(root, "seg", f"{case_name}_seg.npy") for case_name in datalist
        ]
        self.names = datalist
        self.transforms = eval(transforms or "Identity()")
        self.num_classes = num_classes
        self.modal_ind = get_modal_indices(modal)

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        y = np.load(self.segpaths[index])
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        _, h, w, z = y.shape
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_classes, dtype=np.float32)[y]
        yo = np.reshape(one_hot_targets, (1, h, w, z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(len(MASK_ARRAY), 1)
        mask = torch.squeeze(torch.from_numpy(MASK_ARRAY[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)


class BratsEvalNpyDataset(Dataset):
    def __init__(
        self,
        transforms: str = "",
        root: Optional[str] = None,
        split_root: Optional[str] = None,
        split_file: str = "val.csv",
        modal: str = "all",
        num_classes: int = 4,
    ) -> None:
        split_path = resolve_split_path(split_file, split_root)
        dataframe = pd.read_csv(split_path)
        datalist = dataframe["case"]

        self.volpaths = [
            os.path.join(root, "vol", f"{case_name}_vol.npy") for case_name in datalist
        ]
        self.segpaths = [
            os.path.join(root, "seg", f"{case_name}_seg.npy") for case_name in datalist
        ]
        self.names = datalist
        self.transforms = eval(transforms or "Identity()")
        self.num_classes = num_classes
        self.modal_ind = get_modal_indices(modal)
        self.masks = dataframe["mask"].apply(ast.literal_eval)

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        y = np.load(self.segpaths[index]).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        _, h, w, z = y.shape
        y_flat = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_classes, dtype=np.float32)[y_flat]
        yo = np.reshape(one_hot_targets, (1, h, w, z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        mask = torch.squeeze(torch.from_numpy(np.array(self.masks[index])), dim=0)
        return x, y, mask, yo, name

    def __len__(self):
        return len(self.volpaths)
