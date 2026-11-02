"""Paper configuration used for the released PRISM implementation."""

from copy import deepcopy

GPU_PRESET_CHOICES = ("paper",)

_PAPER_PRESET = {
    "batch_size": 2,
    "lr": 3e-4,
    "weight_decay": 3e-5,
    "num_epochs": 800,
    "num_workers": 4,
    "val_every": 25,
    "crop_size": [128, 128, 128],
    "patch_size": 128,
    "base_channels": 16,
    "bottleneck_channels": 128,
    "latent_rank": 16,
    "pooled_size": 4,
    "use_checkpoint": False,
    "amp": True,
    "tf32": True,
    "cudnn_benchmark": True,
}

TRAIN_PRESETS = {"paper": _PAPER_PRESET}
TEST_PRESETS = {
    "paper": {
        key: value
        for key, value in _PAPER_PRESET.items()
        if key
        not in {
            "batch_size",
            "lr",
            "weight_decay",
            "num_epochs",
            "num_workers",
            "val_every",
            "crop_size",
            "amp",
        }
    }
}


def get_train_preset(name: str):
    if name not in TRAIN_PRESETS:
        raise ValueError(f"Unsupported preset: {name}")
    return deepcopy(TRAIN_PRESETS[name])


def get_test_preset(name: str):
    if name not in TEST_PRESETS:
        raise ValueError(f"Unsupported preset: {name}")
    return deepcopy(TEST_PRESETS[name])
