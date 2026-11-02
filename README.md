# PRISM

Implementation of **PRISM: Lightweight Prior-Guided Segmentation of Incomplete Multimodal Brain Tumor MRI**.

PRISM segments brain tumours from any non-empty subset of four MRI modalities. It combines a Progressive Shared Encoder (PSE), Mask-Aware Modal Fusion (MAMF), Cross-Scale Context Aggregation (CSCA), Lightweight Prior Bottleneck (LPB), and Prior-Guided Multi-Source Decoder (PGMSD).

## Installation

Use Python 3.10 or newer. Install a PyTorch build suitable for your hardware, then install the dependencies:

```bash
pip install -r requirements.txt
```

Training supports CUDA and automatic mixed precision. CPU execution is available for small development checks. Weights & Biases logging is optional (`pip install wandb`) and disabled by default.

## Data preparation

Obtain the BraTS2021 or BraTS2023 Adult Glioma data through the official providers and follow their access and use terms. Keep raw images and all generated files outside this repository.

The expected modality order throughout the code is **FLAIR, T1ce, T1, T2**. BraTS2021 files use the `_flair`, `_t1ce`, `_t1`, `_t2`, `_seg` suffixes (the alternative `_brain_*` and `_final_seg` names are also supported). BraTS2023 GLI uses `-t2f`, `-t1c`, `-t1n`, `-t2w`, `-seg`. The preprocessing script discovers case directories recursively, crops the foreground, normalizes intensities, maps label 4 to internal label 3, and pads to at least 128 voxels per spatial axis.

```bash
python preprocess.py \
  --dataname BRATS2023 \
  --input-path /path/to/raw_brats2023 \
  --output-path /path/to/local_data/brats2023 \
  --split-output-root /path/to/local_splits/brats2023
```

For BraTS2021, use `--dataname BRATS2021` with separate input, output and split directories. Run preprocessing separately for each dataset; their cases are never pooled. Generated arrays are `vol/<CASE_ID>_vol.npy` with shape `(H,W,D,4)` and `seg/<CASE_ID>_seg.npy` with shape `(H,W,D)`.

## Training

```bash
python train.py \
  --dataname BRATS2023 \
  --datapath /path/to/local_data/brats2023 \
  --split_root /path/to/local_splits/brats2023 \
  --savepath /path/to/local_outputs/prism_brats2023
```

The `paper` preset uses 128 x 128 x 128 crops and inference windows, batch size 2, base channels 16, bottleneck channels 128, rank 16, AdamW with learning rate `3e-4` and weight decay `3e-5`, 800 epochs and validation every 25 epochs. The loss is final segmentation loss + `0.4 * prior` + `0.3 * auxiliary` + `0.1 * consistency`. The best mean validation DSC determines `best.pth`. Test evaluation during training is disabled by default.

## Evaluation

```bash
python test.py \
  --dataname BRATS2023 \
  --datapath /path/to/local_data/brats2023 \
  --split_root /path/to/local_splits/brats2023 \
  --resume /path/to/local_outputs/prism_brats2023/best.pth \
  --output_dir /path/to/local_outputs/prism_test
```

Evaluation uses sliding windows with 50% overlap, averages overlapping probabilities and reports DSC and HD95 for whole tumour (WT), tumour core (TC) and enhancing tumour (ET). It averages across cases for each pattern, then averages the 15 pattern means equally. The `pattern_metrics.csv` and `summary.txt` files are generated locally. HD95 assumes 1 mm isotropic voxels by default; set `--voxel_spacing x y z` if required. WT uses internal labels 1/2/3, TC uses 1/3 and ET uses 3.

Checkpoints must match the model defined in this release. Private development checkpoints can contain parameters for removed modules and are not guaranteed to load strictly. No weights are downloaded automatically.

## Repository contents

| Files | Purpose |
| --- | --- |
| `model.py`, `modules.py`, `model_factory.py` | Complete PRISM architecture and checkpoint loading |
| `preprocess.py`, `datasets.py`, `data/` | Preprocessing and data loading |
| `train.py`, `losses.py`, `engine.py`, `presets.py` | Training, objective and runtime configuration |
| `test.py`, `evaluate.py` | Evaluation and sliding-window inference |

## License

The code is distributed under the [MIT License](LICENSE), retaining the copyright notice from the existing release. Dataset access and use are governed separately by the dataset providers.
