import argparse
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from .data.data_utils import init_fn
    from .datasets import (
        BratsEvalNpyDataset,
        BratsTrainNpyDataset,
        build_default_test_transforms,
        build_default_train_transforms,
    )
    from .engine import (
        MultiEpochsDataLoader,
        PolyLRScheduler,
        core_model,
        count_parameters,
        get_num_classes,
        setup_logging,
        setup_seed,
    )
    from .evaluate import test_softmax
    from .losses import compute_prism_loss
    from .model_factory import create_model
    from .presets import GPU_PRESET_CHOICES, get_train_preset
except ImportError:
    from data.data_utils import init_fn
    from datasets import (
        BratsEvalNpyDataset,
        BratsTrainNpyDataset,
        build_default_test_transforms,
        build_default_train_transforms,
    )
    from engine import (
        MultiEpochsDataLoader,
        PolyLRScheduler,
        core_model,
        count_parameters,
        get_num_classes,
        setup_logging,
        setup_seed,
    )
    from evaluate import test_softmax
    from losses import compute_prism_loss
    from model_factory import create_model
    from presets import GPU_PRESET_CHOICES, get_train_preset


MODALITY_NAMES = ["flair", "t1ce", "t1", "t2"]


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--gpu_preset", default="paper", choices=GPU_PRESET_CHOICES)
    pre_args, _ = pre_parser.parse_known_args()
    preset = get_train_preset(pre_args.gpu_preset)

    parser = argparse.ArgumentParser(
        description="Train PRISM on BraTS-style .npy volumes"
    )
    parser.add_argument(
        "--gpu_preset", default=pre_args.gpu_preset, choices=GPU_PRESET_CHOICES
    )
    parser.add_argument("--batch_size", default=preset["batch_size"], type=int)
    parser.add_argument("--datapath", required=True, type=str)
    parser.add_argument("--dataname", default="BRATS2023", type=str)
    parser.add_argument("--savepath", required=True, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--lr", default=preset["lr"], type=float)
    parser.add_argument("--weight_decay", default=preset["weight_decay"], type=float)
    parser.add_argument("--num_epochs", default=preset["num_epochs"], type=int)
    parser.add_argument("--num_workers", default=preset["num_workers"], type=int)
    parser.add_argument("--val_every", default=preset["val_every"], type=int)
    parser.add_argument("--seed", default=999, type=int)
    parser.add_argument("--split_root", required=True, type=str)
    parser.add_argument("--train_file", default="train.txt", type=str)
    parser.add_argument("--val_file", default="val.csv", type=str)
    parser.add_argument("--test_file", default="test.csv", type=str)
    parser.add_argument("--crop_size", nargs=3, default=preset["crop_size"], type=int)
    parser.add_argument("--patch_size", default=preset["patch_size"], type=int)
    parser.add_argument("--train_transforms", default=None, type=str)
    parser.add_argument("--test_transforms", default=None, type=str)
    parser.add_argument("--base_channels", default=preset["base_channels"], type=int)
    parser.add_argument(
        "--bottleneck_channels", default=preset["bottleneck_channels"], type=int
    )
    parser.add_argument("--latent_rank", default=preset["latent_rank"], type=int)
    parser.add_argument("--pooled_size", default=preset["pooled_size"], type=int)
    parser.add_argument("--norm", default="gn", choices=["gn", "in", "bn"])
    parser.set_defaults(
        use_checkpoint=preset["use_checkpoint"],
        amp=preset["amp"],
        tf32=preset["tf32"],
        cudnn_benchmark=preset["cudnn_benchmark"],
    )
    parser.add_argument("--use_checkpoint", dest="use_checkpoint", action="store_true")
    parser.add_argument(
        "--no_use_checkpoint", dest="use_checkpoint", action="store_false"
    )
    parser.add_argument("--aux_weight", default=0.3, type=float)
    parser.add_argument("--prior_weight", default=0.4, type=float)
    parser.add_argument("--consistency_weight", default=0.1, type=float)
    parser.add_argument("--run_test", action="store_true", default=False)
    parser.add_argument("--log_every", default=1, type=int)
    parser.add_argument("--wandb_project", default="PRISM", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--wandb_id", default=None, type=str)
    parser.add_argument(
        "--wandb_mode", default="disabled", choices=["online", "offline", "disabled"]
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--tf32", dest="tf32", action="store_true")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument(
        "--cudnn_benchmark", dest="cudnn_benchmark", action="store_true"
    )
    parser.add_argument(
        "--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false"
    )
    args = parser.parse_args()
    return args


def configure_cuda_backend(args, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark) and not bool(
            args.debug
        )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")


def prepare_output_dir(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    try:
        os.makedirs(resolved, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create savepath: {resolved}. "
            "This path is not writable by the current user. "
            "Use a user-owned directory such as './outputs/prism_smoke' or '/data/<user>/outputs/prism_smoke'."
        ) from exc
    return resolved


def create_dataloaders(args, num_classes: int, device: torch.device):
    train_set = BratsTrainNpyDataset(
        transforms=args.train_transforms,
        root=args.datapath,
        split_root=args.split_root,
        num_classes=num_classes,
        train_file=args.train_file,
    )
    val_set = BratsEvalNpyDataset(
        transforms=args.test_transforms,
        root=args.datapath,
        split_root=args.split_root,
        num_classes=num_classes,
        split_file=args.val_file,
    )
    test_set = BratsEvalNpyDataset(
        transforms=args.test_transforms,
        root=args.datapath,
        split_root=args.split_root,
        num_classes=num_classes,
        split_file=args.test_file,
    )

    pin_memory = device.type == "cuda"
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        worker_init_fn=init_fn,
    )
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def move_batch_to_device(batch, device: torch.device):
    x, target, mask = batch[:3]
    x = x.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    return x, target, mask


def log_epoch_stats(epoch: int, stats: Dict[str, float], lr: float) -> None:
    msg = (
        f"Epoch {epoch}: loss={stats['loss']:.4f}, final={stats['final_loss']:.4f}, "
        f"prior={stats['prior_loss']:.4f}, aux={stats['aux_loss']:.4f}, "
        f"cons={stats['consistency']:.4f}, lr={lr:.8f}"
    )
    logging.info(msg)


def build_wandb_run_name(args) -> str:
    if args.wandb_name:
        return args.wandb_name
    crop_name = "x".join(str(item) for item in args.crop_size)
    job_id = os.getenv("SLURM_JOB_ID")
    name = (
        f"{args.dataname}_prism_{args.gpu_preset}"
        f"_bc{args.base_channels}_bot{args.bottleneck_channels}_crop{crop_name}_patch{args.patch_size}"
    )
    if args.use_checkpoint:
        name += "_ckpt"
    if job_id:
        name += f"_job{job_id}"
    return name


def init_wandb(args, num_classes: int, param_count: int):
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed. Install it or run with --wandb_mode disabled."
        )

    run_name = build_wandb_run_name(args)
    run_id = args.wandb_id or run_name
    config = dict(vars(args))
    config["num_classes"] = num_classes
    config["param_count"] = param_count
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        id=run_id,
        mode=args.wandb_mode,
        resume="allow",
        config=config,
    )


def wandb_active() -> bool:
    return wandb is not None and wandb.run is not None


def collect_modal_weight_logs(modal_weights) -> Dict[str, float]:
    if not modal_weights:
        return {}
    logs = {}
    for scale_name, weights in modal_weights.items():
        mean_weights = weights.detach().float().mean(dim=0).cpu()
        for modal_idx, modal_name in enumerate(MODALITY_NAMES[: mean_weights.numel()]):
            logs[f"train_iter/modal_weight/{scale_name}/{modal_name}"] = float(
                mean_weights[modal_idx]
            )
    return logs


def get_memory_logs(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated(device)
        / (1024**3),
        "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "system/gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated(device)
        / (1024**3),
    }


def get_segmentation_class_labels(num_classes: int) -> Dict[int, str]:
    if num_classes == 4:
        return {0: "background", 1: "ncr_net", 2: "edema", 3: "enhancing"}
    return {
        0: "background",
        1: "necrosis",
        2: "edema",
        3: "non_enhancing",
        4: "enhancing",
    }


def normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    image_slice = image_slice.astype(np.float32)
    image_slice = image_slice - image_slice.min()
    denom = max(float(image_slice.max()), 1e-6)
    image_slice = image_slice / denom
    return (image_slice * 255.0).astype(np.uint8)


def build_validation_preview(
    model, val_loader, device: torch.device, patch_size: int, num_classes: int
):
    if not wandb_active():
        return None

    batch = next(iter(val_loader))
    x = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True)
    mask = batch[2].to(device, non_blocking=True)
    names = batch[-1]

    model_core = core_model(model)
    previous_training_flag = model_core.is_training
    previous_mode = model.training
    model_core.is_training = False
    model.eval()
    with torch.no_grad():
        pred = model(x, mask)
    if previous_mode:
        model.train()
    model_core.is_training = previous_training_flag

    pred_label = torch.argmax(pred, dim=1)
    tumor_pixels = (target[0] > 0).float().sum(dim=(0, 1))
    slice_idx = (
        int(torch.argmax(tumor_pixels).item())
        if float(tumor_pixels.max().item()) > 0
        else int(target.shape[-1] // 2)
    )

    image_slice = normalize_slice(x[0, 0, :, :, slice_idx].detach().cpu().numpy())
    pred_slice = pred_label[0, :, :, slice_idx].detach().cpu().numpy().astype(np.int32)
    target_slice = target[0, :, :, slice_idx].detach().cpu().numpy().astype(np.int32)
    class_labels = get_segmentation_class_labels(num_classes)
    caption = f"{names[0]} | mask={mask[0].int().cpu().tolist()} | z={slice_idx} | patch={patch_size}"
    return wandb.Image(
        image_slice,
        caption=caption,
        masks={
            "prediction": {"mask_data": pred_slice, "class_labels": class_labels},
            "ground_truth": {"mask_data": target_slice, "class_labels": class_labels},
        },
    )


def main():
    args = parse_args()
    if args.train_transforms is None:
        args.train_transforms = build_default_train_transforms(tuple(args.crop_size))
    if args.test_transforms is None:
        args.test_transforms = build_default_test_transforms()
    args.savepath = prepare_output_dir(args.savepath)
    setup_logging(args.savepath)
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_backend(args, device)

    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    num_classes = get_num_classes(args.dataname)

    model = create_model(args, num_classes=num_classes)
    param_count = count_parameters(model)
    logging.info("PRISM trainable params: %.2f M", param_count / 1e6)
    init_wandb(args, num_classes=num_classes, param_count=param_count)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = PolyLRScheduler(args.lr, args.num_epochs)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    train_loader, val_loader, test_loader = create_dataloaders(
        args, num_classes, device
    )
    train_iter = iter(train_loader)

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optim_dict"])
        if "scaler_dict" in checkpoint and checkpoint["scaler_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", best_dice)
        logging.info(
            "Resumed from epoch %d with best_dice %.4f", start_epoch, best_dice
        )

    total_start = time.time()
    try:
        for epoch in range(start_epoch, args.num_epochs):
            epoch_start = time.time()
            lr = scheduler(optimizer, epoch)
            model.train()
            core_model(model).is_training = True
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            epoch_stats = {
                "loss": 0.0,
                "final_loss": 0.0,
                "prior_loss": 0.0,
                "aux_loss": 0.0,
                "consistency": 0.0,
            }
            iter_per_epoch = len(train_loader)

            for iteration in range(iter_per_epoch):
                iter_start = time.time()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                x, target, mask = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=scaler.is_enabled()):
                    outputs = model(x, mask)
                    total_loss, stats = compute_prism_loss(
                        outputs,
                        target,
                        num_classes=num_classes,
                        aux_weight=args.aux_weight,
                        prior_weight=args.prior_weight,
                        consistency_weight=args.consistency_weight,
                    )

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                for key in epoch_stats:
                    epoch_stats[key] += stats[key].item()

                global_step = epoch * iter_per_epoch + iteration + 1
                if iteration % args.log_every == 0:
                    iter_time = time.time() - iter_start
                    logging.info(
                        "Epoch %d/%d Iter %d/%d loss=%.4f final=%.4f prior=%.4f aux=%.4f cons=%.4f lr=%.8f time=%.2fs",
                        epoch + 1,
                        args.num_epochs,
                        iteration + 1,
                        iter_per_epoch,
                        stats["loss"].item(),
                        stats["final_loss"].item(),
                        stats["prior_loss"].item(),
                        stats["aux_loss"].item(),
                        stats["consistency"].item(),
                        lr,
                        iter_time,
                    )
                    if wandb_active():
                        train_log = {
                            "train_iter/epoch": epoch + 1,
                            "train_iter/loss": stats["loss"].item(),
                            "train_iter/final_loss": stats["final_loss"].item(),
                            "train_iter/final_cross": stats["final_cross"].item(),
                            "train_iter/final_dice": stats["final_dice"].item(),
                            "train_iter/prior_loss": stats["prior_loss"].item(),
                            "train_iter/prior_cross": stats["prior_cross"].item(),
                            "train_iter/prior_dice": stats["prior_dice"].item(),
                            "train_iter/aux_loss": stats["aux_loss"].item(),
                            "train_iter/aux_cross": stats["aux_cross"].item(),
                            "train_iter/aux_dice": stats["aux_dice"].item(),
                            "train_iter/consistency": stats["consistency"].item(),
                            "train_iter/lr": lr,
                            "train_iter/present_modalities": mask.float()
                            .sum(dim=1)
                            .mean()
                            .item(),
                            "train_iter/iter_time_sec": iter_time,
                        }
                        train_log.update(
                            collect_modal_weight_logs(outputs.get("modal_weights", {}))
                        )
                        train_log.update(get_memory_logs(device))
                        wandb.log(train_log, step=global_step)

                if args.debug and iteration >= 1:
                    break

            effective_iters = 2 if args.debug else iter_per_epoch
            for key in epoch_stats:
                epoch_stats[key] /= float(max(effective_iters, 1))
            log_epoch_stats(epoch + 1, epoch_stats, lr)
            epoch_minutes = (time.time() - epoch_start) / 60.0
            logging.info("Epoch %d finished in %.2f min", epoch + 1, epoch_minutes)

            if wandb_active():
                wandb.log(
                    {
                        "train/epoch": epoch + 1,
                        "train/loss": epoch_stats["loss"],
                        "train/final_loss": epoch_stats["final_loss"],
                        "train/prior_loss": epoch_stats["prior_loss"],
                        "train/aux_loss": epoch_stats["aux_loss"],
                        "train/consistency": epoch_stats["consistency"],
                        "train/learning_rate": lr,
                        "train/epoch_time_min": epoch_minutes,
                    },
                    step=(epoch + 1) * iter_per_epoch,
                )

            last_path = os.path.join(args.savepath, "model_last.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "scaler_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_dice": best_dice,
                    "args": vars(args),
                },
                last_path,
            )

            should_validate = (
                ((epoch + 1) % args.val_every == 0)
                or (epoch + 1 == args.num_epochs)
                or args.debug
            )
            if should_validate:
                with torch.no_grad():
                    val_scores, val_loss = test_softmax(
                        val_loader,
                        model,
                        dataname=args.dataname,
                        patch_size=args.patch_size,
                    )
                val_wt, val_tc, val_et, val_etpp = val_scores
                val_dice = float((val_wt + val_tc + val_et) / 3.0)
                logging.info(
                    "Validation epoch=%d WT=%.4f TC=%.4f ET=%.4f ETpp=%.4f loss=%.4f mean=%.4f",
                    epoch + 1,
                    float(val_wt),
                    float(val_tc),
                    float(val_et),
                    float(val_etpp),
                    float(val_loss),
                    val_dice,
                )

                if wandb_active():
                    val_log = {
                        "val/epoch": epoch + 1,
                        "val/WT_Dice": float(val_wt),
                        "val/TC_Dice": float(val_tc),
                        "val/ET_Dice": float(val_et),
                        "val/ETpp_Dice": float(val_etpp),
                        "val/mean_Dice": val_dice,
                        "val/seg_loss": float(val_loss),
                    }
                    val_preview = build_validation_preview(
                        model,
                        val_loader,
                        device=device,
                        patch_size=args.patch_size,
                        num_classes=num_classes,
                    )
                    if val_preview is not None:
                        val_log["val/preview"] = val_preview
                    wandb.log(val_log, step=(epoch + 1) * iter_per_epoch)

                if val_dice > best_dice:
                    best_dice = val_dice
                    best_path = os.path.join(args.savepath, "best.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optim_dict": optimizer.state_dict(),
                            "scaler_dict": (
                                scaler.state_dict() if scaler.is_enabled() else None
                            ),
                            "best_dice": best_dice,
                            "args": vars(args),
                        },
                        best_path,
                    )
                    logging.info("Saved new best checkpoint to %s", best_path)
                    if wandb_active():
                        wandb.run.summary["best_dice"] = best_dice
                        wandb.run.summary["best_epoch"] = epoch + 1

                if args.run_test:
                    with torch.no_grad():
                        test_scores, test_loss = test_softmax(
                            test_loader,
                            model,
                            dataname=args.dataname,
                            patch_size=args.patch_size,
                        )
                    test_wt, test_tc, test_et, test_etpp = test_scores
                    logging.info(
                        "Test epoch=%d WT=%.4f TC=%.4f ET=%.4f ETpp=%.4f loss=%.4f",
                        epoch + 1,
                        float(test_wt),
                        float(test_tc),
                        float(test_et),
                        float(test_etpp),
                        float(test_loss),
                    )
                    if wandb_active():
                        wandb.log(
                            {
                                "test/epoch": epoch + 1,
                                "test/WT_Dice": float(test_wt),
                                "test/TC_Dice": float(test_tc),
                                "test/ET_Dice": float(test_et),
                                "test/ETpp_Dice": float(test_etpp),
                                "test/mean_Dice": float(
                                    (test_wt + test_tc + test_et) / 3.0
                                ),
                                "test/seg_loss": float(test_loss),
                            },
                            step=(epoch + 1) * iter_per_epoch,
                        )

                model.train()
                core_model(model).is_training = True

            if args.debug:
                break

        total_hours = (time.time() - total_start) / 3600.0
        logging.info("Total training time: %.2f hours", total_hours)
        if wandb_active():
            wandb.run.summary["total_training_hours"] = total_hours
    finally:
        if wandb_active():
            wandb.finish()


if __name__ == "__main__":
    main()
