from typing import Dict, Tuple

import torch

try:
    from .engine import core_model
    from .model import PRISMFuse
except ImportError:
    from engine import core_model
    from model import PRISMFuse


MODEL_ARG_FIELDS = (
    "base_channels",
    "bottleneck_channels",
    "latent_rank",
    "pooled_size",
    "norm",
    "use_checkpoint",
)


def create_model(args, num_classes: int) -> torch.nn.Module:
    return PRISMFuse(
        num_classes=num_classes,
        base_channels=args.base_channels,
        bottleneck_channels=args.bottleneck_channels,
        latent_rank=args.latent_rank,
        pooled_size=args.pooled_size,
        norm=args.norm,
        use_checkpoint=args.use_checkpoint,
    )


def _strip_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key if key.startswith("module.") else f"module.{key}"): value
        for key, value in state_dict.items()
    }


def load_checkpoint_state(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_state = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )

    candidates = [raw_state]
    stripped_state = _strip_module_prefix(raw_state)
    if stripped_state.keys() != raw_state.keys():
        candidates.append(stripped_state)
    module_state = _add_module_prefix(raw_state)
    if module_state.keys() != raw_state.keys():
        candidates.append(module_state)

    last_error = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return checkpoint
        except RuntimeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")


def load_checkpoint_file(checkpoint_path: str, device: torch.device):
    return torch.load(checkpoint_path, map_location=device)


def inject_model_args_from_checkpoint(args, checkpoint) -> None:
    if not isinstance(checkpoint, dict):
        return
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        return
    for field in MODEL_ARG_FIELDS:
        if field in saved_args:
            setattr(args, field, saved_args[field])


def build_eval_model(
    args, num_classes: int, device: torch.device
) -> Tuple[torch.nn.Module, Dict]:
    checkpoint = load_checkpoint_file(args.resume, device)
    inject_model_args_from_checkpoint(args, checkpoint)
    model = create_model(args, num_classes=num_classes)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    raw_state = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    candidates = [raw_state]
    stripped_state = _strip_module_prefix(raw_state)
    if stripped_state.keys() != raw_state.keys():
        candidates.append(stripped_state)
    module_state = _add_module_prefix(raw_state)
    if module_state.keys() != raw_state.keys():
        candidates.append(module_state)

    last_error = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    model_core = core_model(model)
    model_core.is_training = False
    model.eval()
    return model, checkpoint
