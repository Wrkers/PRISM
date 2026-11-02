"""Evaluate PRISM under the 15 BraTS modality-availability patterns.

The output follows the P1-P14 plus Full ordering used in the manuscript and
reports pattern-balanced DSC and HD95 for WT, TC, and ET.
"""

import argparse
import contextlib
import csv
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from torch.utils.data import DataLoader

from datasets import BratsEvalNpyDataset, build_default_test_transforms
from engine import get_num_classes
from model_factory import build_eval_model

MODALITY_ORDER = ("FLAIR", "T1ce", "T1", "T2")
REGIONS = ("WT", "TC", "ET")
PAPER_PATTERNS = (
    ("P1", "T1", (1, 1, 0, 1)),
    ("P2", "T2", (1, 1, 1, 0)),
    ("P3", "T1ce", (1, 0, 1, 1)),
    ("P4", "FLAIR", (0, 1, 1, 1)),
    ("P5", "T1/T2", (1, 1, 0, 0)),
    ("P6", "T1/T1ce", (1, 0, 0, 1)),
    ("P7", "T1/FLAIR", (0, 1, 0, 1)),
    ("P8", "T2/T1ce", (1, 0, 1, 0)),
    ("P9", "T2/FLAIR", (0, 1, 1, 0)),
    ("P10", "T1ce/FLAIR", (0, 0, 1, 1)),
    ("P11", "T1/T2/T1ce", (1, 0, 0, 0)),
    ("P12", "T1/T2/FLAIR", (0, 1, 0, 0)),
    ("P13", "T1/T1ce/FLAIR", (0, 0, 0, 1)),
    ("P14", "T2/T1ce/FLAIR", (0, 0, 1, 0)),
    ("Full", "None", (1, 1, 1, 1)),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PRISM across all 15 availability patterns."
    )
    parser.add_argument("--dataname", default="BRATS2023")
    parser.add_argument(
        "--datapath",
        required=True,
        help="Preprocessed data root containing vol/ and seg/.",
    )
    parser.add_argument(
        "--split_root", required=True, help="Directory containing the fixed test CSV."
    )
    parser.add_argument("--test_file", default="test.csv")
    parser.add_argument(
        "--resume", required=True, help="Checkpoint created by train.py."
    )
    parser.add_argument("--output_dir", default="outputs/test")
    parser.add_argument("--patch_size", default=128, type=int)
    parser.add_argument("--overlap", default=0.5, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_cases", default=None, type=int)
    parser.add_argument("--base_channels", default=16, type=int)
    parser.add_argument("--bottleneck_channels", default=128, type=int)
    parser.add_argument("--latent_rank", default=16, type=int)
    parser.add_argument("--pooled_size", default=4, type=int)
    parser.add_argument("--norm", default="gn", choices=("gn", "in", "bn"))
    parser.add_argument("--voxel_spacing", nargs=3, default=(1.0, 1.0, 1.0), type=float)
    parser.add_argument(
        "--amp", action="store_true", help="Use CUDA automatic mixed precision."
    )
    parser.add_argument("--use_checkpoint", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.overlap < 1.0:
        parser.error("--overlap must be in (0, 1).")
    return args


def load_model(args, device: torch.device):
    """Load a training checkpoint while restoring its recorded model settings."""
    model, _checkpoint = build_eval_model(args, get_num_classes(args.dataname), device)
    model.eval()
    return model


def window_starts(length: int, patch_size: int, overlap: float) -> Iterable[int]:
    if length < patch_size:
        raise ValueError(
            f"Volume dimension {length} is smaller than patch size {patch_size}."
        )
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def sliding_window_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    patch_size: int,
    overlap: float,
    amp: bool,
) -> torch.Tensor:
    _, _, height, width, depth = volume.shape
    h_starts = window_starts(height, patch_size, overlap)
    w_starts = window_starts(width, patch_size, overlap)
    d_starts = window_starts(depth, patch_size, overlap)
    probabilities = torch.zeros(
        (1, num_classes, height, width, depth), device=volume.device
    )
    counts = torch.zeros((1, 1, height, width, depth), device=volume.device)
    autocast_context = (
        torch.autocast("cuda", enabled=amp)
        if volume.device.type == "cuda"
        else contextlib.nullcontext()
    )

    with torch.inference_mode():
        for hs in h_starts:
            for ws in w_starts:
                for ds in d_starts:
                    patch = volume[
                        :,
                        :,
                        hs : hs + patch_size,
                        ws : ws + patch_size,
                        ds : ds + patch_size,
                    ]
                    with autocast_context:
                        patch_probabilities = model(patch, mask)
                    probabilities[
                        :,
                        :,
                        hs : hs + patch_size,
                        ws : ws + patch_size,
                        ds : ds + patch_size,
                    ] += patch_probabilities.float()
                    counts[
                        :,
                        :,
                        hs : hs + patch_size,
                        ws : ws + patch_size,
                        ds : ds + patch_size,
                    ] += 1
    return torch.argmax(probabilities / counts.clamp_min(1), dim=1)


def region_masks(label: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "WT": label > 0,
        "TC": np.logical_or(label == 1, label == 3),
        "ET": label == 3,
    }


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(prediction.sum() + target.sum())
    if denominator == 0.0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def hd95(prediction: np.ndarray, target: np.ndarray, spacing: Sequence[float]) -> float:
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(
            np.linalg.norm((np.asarray(prediction.shape) - 1) * np.asarray(spacing))
        )
    structure = generate_binary_structure(prediction.ndim, 1)
    prediction_surface = np.logical_xor(
        prediction, binary_erosion(prediction, structure=structure, border_value=0)
    )
    target_surface = np.logical_xor(
        target, binary_erosion(target, structure=structure, border_value=0)
    )
    prediction_distance = distance_transform_edt(~prediction_surface, sampling=spacing)
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    distances = np.concatenate(
        (target_distance[prediction_surface], prediction_distance[target_surface])
    )
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def case_metrics(
    prediction: torch.Tensor, target: torch.Tensor, spacing: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    prediction_regions = region_masks(prediction[0].cpu().numpy())
    target_regions = region_masks(target[0].cpu().numpy())
    dsc = np.asarray(
        [dice(prediction_regions[region], target_regions[region]) for region in REGIONS]
    )
    hd = np.asarray(
        [
            hd95(prediction_regions[region], target_regions[region], spacing)
            for region in REGIONS
        ]
    )
    return dsc, hd


def evaluate_pattern(args, loader, model, device, pattern) -> Dict[str, object]:
    pattern_id, missing, binary_mask = pattern
    num_classes = get_num_classes(args.dataname)
    mask = torch.tensor(binary_mask, device=device, dtype=torch.float32).unsqueeze(0)
    dsc_values, hd95_values = [], []
    for volume, target, _csv_mask, _one_hot, _name in loader:
        volume = volume.to(device=device, dtype=torch.float32, non_blocking=True)
        prediction = sliding_window_inference(
            model,
            volume,
            mask,
            num_classes,
            args.patch_size,
            args.overlap,
            args.amp and device.type == "cuda",
        )
        dsc, hd = case_metrics(prediction, target, args.voxel_spacing)
        dsc_values.append(dsc)
        hd95_values.append(hd)
    return {
        "pattern": pattern_id,
        "missing": missing,
        "mask": "[{}]".format(", ".join(str(value) for value in binary_mask)),
        "dsc": np.mean(np.stack(dsc_values), axis=0),
        "hd95": np.mean(np.stack(hd95_values), axis=0),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BratsEvalNpyDataset(
        transforms=build_default_test_transforms(),
        root=args.datapath,
        split_root=args.split_root,
        split_file=args.test_file,
        num_classes=get_num_classes(args.dataname),
    )
    if args.max_cases is not None:
        dataset.volpaths, dataset.segpaths, dataset.names, dataset.masks = (
            dataset.volpaths[: args.max_cases],
            dataset.segpaths[: args.max_cases],
            dataset.names[: args.max_cases],
            dataset.masks[: args.max_cases],
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args, device)
    reports = []
    for pattern in PAPER_PATTERNS:
        report = evaluate_pattern(args, loader, model, device, pattern)
        reports.append(report)
        print(
            f"{report['pattern']:>4} | DSC={report['dsc'].mean():.4f} | HD95={report['hd95'].mean():.4f}"
        )

    all_dsc = np.mean(np.stack([report["dsc"] for report in reports]), axis=0)
    all_hd95 = np.mean(np.stack([report["hd95"] for report in reports]), axis=0)
    with (output_dir / "pattern_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "pattern",
                "missing_modalities",
                "availability_mask",
                "WT_DSC",
                "TC_DSC",
                "ET_DSC",
                "WT_HD95",
                "TC_HD95",
                "ET_HD95",
            )
        )
        for report in reports:
            writer.writerow(
                (
                    report["pattern"],
                    report["missing"],
                    report["mask"],
                    *report["dsc"],
                    *report["hd95"],
                )
            )
        writer.writerow(("All-15", "pattern-balanced mean", "-", *all_dsc, *all_hd95))
    summary = f"All-15 mean DSC: {all_dsc.mean():.4f}\nAll-15 mean HD95: {all_hd95.mean():.4f}\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
