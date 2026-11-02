import csv
import os
import random
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import medpy.io as medio
except ImportError:  # pragma: no cover
    medio = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


MODALITY_ORDER = ("flair", "t1ce", "t1", "t2", "seg")
MASK_ARRAY: Sequence[Sequence[bool]] = (
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (False, False, False, True),
    (True, True, False, False),
    (True, False, True, False),
    (True, False, False, True),
    (False, True, True, False),
    (False, True, False, True),
    (False, False, True, True),
    (True, True, True, False),
    (True, True, False, True),
    (True, False, True, True),
    (False, True, True, True),
    (True, True, True, True),
)
DATASET_CONFIGS = {
    "BRATS2021": {
        "case_patterns": (re.compile(r"^BraTS2021_\d+$", re.IGNORECASE),),
        "modality_patterns": {
            "flair": ("_brain_flair.nii.gz", "_flair.nii.gz"),
            "t1ce": ("_brain_t1ce.nii.gz", "_t1ce.nii.gz"),
            "t1": ("_brain_t1.nii.gz", "_t1.nii.gz"),
            "t2": ("_brain_t2.nii.gz", "_t2.nii.gz"),
            "seg": ("_final_seg.nii.gz", "_seg.nii.gz"),
        },
    },
    "BRATS2023": {
        "case_patterns": (
            re.compile(r"^BraTS-GLI-\d{5}-\d{3}$", re.IGNORECASE),
            re.compile(r"^BraTS2023_\d+$", re.IGNORECASE),
        ),
        "modality_patterns": {
            "flair": ("-t2f.nii.gz",),
            "t1ce": ("-t1c.nii.gz",),
            "t1": ("-t1n.nii.gz",),
            "t2": ("-t2w.nii.gz",),
            "seg": ("-seg.nii.gz",),
        },
    },
}


def sup_128(xmin: int, xmax: int) -> Tuple[int, int]:
    if xmax - xmin < 128:
        print("#" * 100)
        ecart = int((128 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop(vol: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)
    if x_nonzeros.size == 0:
        x_dim, y_dim, z_dim = tuple(vol.shape)
        return 0, x_dim, 0, y_dim, 0, z_dim

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(int(x_min), int(x_max))
    y_min, y_max = sup_128(int(y_min), int(y_max))
    z_min, z_max = sup_128(int(z_min), int(z_max))

    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(vol: np.ndarray) -> np.ndarray:
    mask = vol.sum(0) > 0
    if not np.any(mask):
        return vol

    for modal_idx in range(vol.shape[0]):
        x = vol[modal_idx, ...]
        y = x[mask]
        if y.size == 0:
            continue
        std = float(y.std())
        if std < 1e-8:
            vol[modal_idx, ...] = x - float(y.mean())
            continue
        vol[modal_idx, ...] = (x - float(y.mean())) / std
    return vol


def pad_to_min_size(
    vol: np.ndarray,
    seg: np.ndarray,
    min_size: Sequence[int] = (128, 128, 128),
) -> Tuple[np.ndarray, np.ndarray]:
    if len(vol.shape) != 4:
        raise ValueError(f"Expected vol shape (C,H,W,D), got {vol.shape}")
    if len(seg.shape) != 3:
        raise ValueError(f"Expected seg shape (H,W,D), got {seg.shape}")

    spatial_shape = vol.shape[1:]
    if seg.shape != spatial_shape:
        raise ValueError(
            f"vol spatial shape {spatial_shape} does not match seg shape {seg.shape}"
        )

    pad_widths = []
    for current, target in zip(spatial_shape, min_size):
        total_pad = max(int(target) - int(current), 0)
        before = total_pad // 2
        after = total_pad - before
        pad_widths.append((before, after))

    if not any(before or after for before, after in pad_widths):
        return vol, seg

    vol = np.pad(
        vol,
        ((0, 0), pad_widths[0], pad_widths[1], pad_widths[2]),
        mode="constant",
        constant_values=0,
    )
    seg = np.pad(
        seg,
        (pad_widths[0], pad_widths[1], pad_widths[2]),
        mode="constant",
        constant_values=0,
    )
    return vol, seg


def load_volume(path: Path):
    if medio is not None:
        return medio.load(str(path))
    if nib is None:
        raise ImportError(
            "Neither medpy nor nibabel is installed, cannot load NIfTI files."
        )
    image = nib.load(str(path))
    return np.asarray(image.get_fdata()), image.header


def build_dataset_config(dataname: str) -> Dict[str, object]:
    if dataname not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataname: {dataname}")
    return DATASET_CONFIGS[dataname]


def is_case_dir(path: Path, case_patterns: Sequence[re.Pattern]) -> bool:
    return any(pattern.match(path.name) for pattern in case_patterns)


def iter_case_dirs(root: Path, case_patterns: Sequence[re.Pattern]) -> Iterable[Path]:
    yielded = set()
    if is_case_dir(root, case_patterns):
        yielded.add(root.resolve())
        yield root

    for current_root, dirnames, _ in os.walk(root):
        current_path = Path(current_root)
        matched_dirs = [
            current_path / dirname
            for dirname in dirnames
            if is_case_dir(current_path / dirname, case_patterns)
        ]
        for case_dir in sorted(matched_dirs):
            resolved = case_dir.resolve()
            if resolved not in yielded:
                yielded.add(resolved)
                yield case_dir
            dirnames.remove(case_dir.name)


def match_modality(path: Path, patterns: Sequence[str]) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(pattern) for pattern in patterns)


def select_best_candidate(paths: List[Path]) -> Path:
    return sorted(paths, key=lambda item: (len(item.parts), len(str(item))))[0]


def resolve_case_files(
    case_dir: Path, modality_patterns: Dict[str, Sequence[str]]
) -> Optional[Dict[str, Path]]:
    files = [path for path in case_dir.rglob("*.nii.gz") if path.is_file()]
    if not files:
        return None

    resolved: Dict[str, Path] = {}
    for modal in MODALITY_ORDER:
        candidates = [
            path for path in files if match_modality(path, modality_patterns[modal])
        ]
        if not candidates:
            return None
        resolved[modal] = select_best_candidate(candidates)
    return resolved


def preprocess_case(
    case_id: str,
    case_files: Dict[str, Path],
    output_root: Path,
    min_size: Sequence[int] = (128, 128, 128),
) -> Tuple[Path, Path]:
    flair, _ = load_volume(case_files["flair"])
    t1ce, _ = load_volume(case_files["t1ce"])
    t1, _ = load_volume(case_files["t1"])
    t2, _ = load_volume(case_files["t2"])

    vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    vol = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])

    seg, _ = load_volume(case_files["seg"])
    seg = seg.astype(np.uint8)
    seg = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    seg[seg == 4] = 3

    vol, seg = pad_to_min_size(vol, seg, min_size=min_size)
    vol = vol.transpose(1, 2, 3, 0)

    vol_path = output_root / "vol" / f"{case_id}_vol.npy"
    seg_path = output_root / "seg" / f"{case_id}_seg.npy"
    np.save(vol_path, vol)
    np.save(seg_path, seg)
    return vol_path, seg_path


def discover_preprocessed_case_names(output_root: Path) -> List[str]:
    vol_dir = output_root / "vol"
    if not vol_dir.exists():
        return []

    case_names = []
    for filename in sorted(os.listdir(vol_dir)):
        if filename.endswith("_vol.npy"):
            case_names.append(filename[: -len("_vol.npy")])
    return case_names


def _compute_split_counts(
    num_cases: int, val_ratio: float, test_ratio: float
) -> Tuple[int, int, int]:
    if num_cases < 3:
        return num_cases, 0, 0

    val_count = max(1, int(round(num_cases * val_ratio)))
    test_count = max(1, int(round(num_cases * test_ratio)))
    train_count = num_cases - val_count - test_count
    if train_count < 1:
        deficit = 1 - train_count
        reduce_val = min(deficit, max(val_count - 1, 0))
        val_count -= reduce_val
        deficit -= reduce_val
        reduce_test = min(deficit, max(test_count - 1, 0))
        test_count -= reduce_test
        train_count = num_cases - val_count - test_count
    return train_count, val_count, test_count


def _mask_for_index(index: int) -> List[bool]:
    return list(MASK_ARRAY[index % len(MASK_ARRAY)])


def write_random_splits(
    case_names: Sequence[str],
    split_output_root: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    split_output_root.mkdir(parents=True, exist_ok=True)
    if len(case_names) != len(set(case_names)):
        raise ValueError("Case identifiers must be unique.")
    if len(case_names) < 3:
        raise ValueError(
            "At least three cases are required for train/validation/test splits."
        )
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Ratios must be positive with val_ratio + test_ratio < 1.")
    shuffled = sorted(case_names)
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, test_count = _compute_split_counts(
        len(shuffled), val_ratio=val_ratio, test_ratio=test_ratio
    )

    train_cases = shuffled[:train_count]
    val_cases = shuffled[train_count : train_count + val_count]
    test_cases = shuffled[
        train_count + val_count : train_count + val_count + test_count
    ]

    with open(split_output_root / "train.txt", "w", encoding="utf-8") as file:
        for case_name in train_cases:
            file.write(f"{case_name}\n")

    for filename, cases in (("val.csv", val_cases), ("test.csv", test_cases)):
        with open(
            split_output_root / filename, "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=["case", "mask"])
            writer.writeheader()
            for index, case_name in enumerate(cases):
                writer.writerow(
                    {"case": case_name, "mask": str(_mask_for_index(index))}
                )


def generate_splits_for_dataset(
    dataname: str,
    output_root: Path,
    split_output_root: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Path:
    case_names = discover_preprocessed_case_names(output_root)
    if not case_names:
        raise RuntimeError(
            f"No preprocessed cases were found under: {output_root / 'vol'}"
        )

    build_dataset_config(dataname)
    write_random_splits(
        case_names,
        split_output_root,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    print(f"[split] generated random splits for {dataname} -> {split_output_root}")
    return split_output_root


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataname", type=str, required=True, choices=sorted(DATASET_CONFIGS.keys())
    )
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split-output-root", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--min-size", nargs=3, default=(128, 128, 128), type=int)
    parser.add_argument("--split-seed", type=int, default=999)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    config = build_dataset_config(args.dataname)
    src_path = Path(args.input_path)
    tar_path = Path(args.output_path)
    split_output_root = (
        Path(args.split_output_root) if args.split_output_root else tar_path / "splits"
    )

    (tar_path / "vol").mkdir(parents=True, exist_ok=True)
    (tar_path / "seg").mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        iter_case_dirs(src_path, config["case_patterns"]), key=lambda item: item.name
    )
    if args.case_limit is not None:
        case_dirs = case_dirs[: args.case_limit]

    if not case_dirs:
        raise RuntimeError(
            f"No case directories were found for {args.dataname} under {src_path}. "
            "Please check the input path and whether the .nii files were converted to .nii.gz."
        )

    print(
        f"Found {len(case_dirs)} case directories under {src_path} for {args.dataname}"
    )
    processed = 0
    skipped = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        vol_out = tar_path / "vol" / f"{case_id}_vol.npy"
        seg_out = tar_path / "seg" / f"{case_id}_seg.npy"
        if args.skip_existing and vol_out.exists() and seg_out.exists():
            print(f"[skip] {case_id} already exists")
            skipped += 1
            continue

        case_files = resolve_case_files(case_dir, config["modality_patterns"])
        if case_files is None:
            raise RuntimeError(
                f"Could not resolve all .nii.gz modalities for case: {case_dir}. "
                f"Expected patterns for {args.dataname}: {config['modality_patterns']}"
            )

        print(f"[process] {case_id}")
        for modal in MODALITY_ORDER:
            print(f"  {modal}: {case_files[modal]}")

        preprocess_case(case_id, case_files, tar_path, min_size=tuple(args.min_size))
        processed += 1

    split_root = generate_splits_for_dataset(
        dataname=args.dataname,
        output_root=tar_path,
        split_output_root=split_output_root,
        seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(
        f"Done. processed={processed}, skipped={skipped}, output={tar_path}, "
        f"split_root={split_root}"
    )


if __name__ == "__main__":
    main()
