# PRISM: Parameters and Reproducibility Settings

Configuration reference for *PRISM: Lightweight Prior-Guided Segmentation of Incomplete Multimodal Brain Tumor MRI*.

**Documentation snapshot:** 2026-11-02. Label/HD95 evidence updated from the reliability audit on the same date. **Status:** audited draft. Unresolved settings are marked **Pending verification**.

This document lists the full reference architecture and separates the archived revision protocol from the original manuscript run. A launch command establishes intended settings, not completion of every planned run. Different model sizes, patch sizes, datasets, and ablations must be documented separately.

## Scope and public-code correspondence

The detailed tables preserve the supplied audit of archived source snapshots and evaluation records. Their source keys, line numbers and hashes identify those archived snapshots; they do not identify the current public files. Pending items remain pending. The saved-record counts below are historical audit evidence, not new measurements made with this public checkout.

The following differences were checked against the public source at commit [`7a8d26b`](https://github.com/Wrkers/PRISM/tree/7a8d26b7af7f8ea772e23997d251c9d8e59bed35):

| Item | Archived documentation | Published implementation |
|---|---|---|
| Model constructor defaults | Base/bottleneck widths 24/192; reference configuration 16/128 | Defaults 16/128 in [model.py](https://github.com/Wrkers/PRISM/blob/7a8d26b7af7f8ea772e23997d251c9d8e59bed35/model.py) |
| Weighted cross-entropy probability clamp | Lower bound 0.005 | Lower bound 1e-7 in [losses.py](https://github.com/Wrkers/PRISM/blob/7a8d26b7af7f8ea772e23997d251c9d8e59bed35/losses.py) |
| HD95 surface extraction | Archived reference re-evaluation: 26-neighbor erosion | 6-neighbor erosion in [test.py](https://github.com/Wrkers/PRISM/blob/7a8d26b7af7f8ea772e23997d251c9d8e59bed35/test.py); both use the `(shape - 1)` volume-diagonal penalty for a single empty mask |
| Configuration controls | Archived training entry includes experimental component switches and a different preset-loading path | Public entry exposes the complete PRISM model and reads the numeric paper preset in [train.py](https://github.com/Wrkers/PRISM/blob/7a8d26b7af7f8ea772e23997d251c9d8e59bed35/train.py) |

The 26-neighbor HD95 audit tables must not be presented as results produced by the 6-neighbor public evaluator. This documentation merge does not change model, loss or evaluation code. Raw audit JSON, patient-level metric files, private source snapshots and training artifacts are not bundled with this document.

## Contents

- [Scope and public-code correspondence](#scope-and-public-code-correspondence)
- [Architecture](#architecture)
- [Training protocol](#training-protocol)
- [Loss functions](#loss-functions)
- [Data processing and inference](#data-processing-and-inference)
- [Label and HD95 verification](#label-and-hd95-verification)
- [Efficiency and software](#efficiency-and-software)
- [Implementation notes and pending verification](#implementation-notes-and-pending-verification)
- [Source evidence](#source-evidence)

**Reference configuration:** encoder widths (16,32,64,96,128); decoder widths (96,64,32,16); bottleneck 128 channels, pooled grid 4³, rank 16; approximately 2.34M trainable parameters. Exact recount and complete original-result provenance remain pending.

**Notation:** C: channel width; DW/PW: depthwise/pointwise convolution; DS: depthwise separable convolution; GN: GroupNorm; GAP: global average pooling. A kernel written as 3³ means 3 × 3 × 3. Keys such as `U:322–349` identify source-file lines in [Source evidence](#source-evidence).

## Architecture

### Inputs, PSE, and basic blocks

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| Input/output | Modalities and classes | 4 modalities in the order FLAIR, T1ce, T1, T2; 4 segmentation classes including background | D:84–94; F:8–23; M:37–38 |
| PSE | Stage widths | 16,32,64,96,128 | F:8–13; M:49–53 |
| PSE | Modality-specific stems | Four independent stems; residual Conv3D 1→16, then stride-2 residual Conv3D 16→32 and a residual depthwise-separable refinement block | M:68–69; U:283–293 |
| PSE | Shared deep stages | Stride-2 residual blocks 32→64→96→128; 1,1,2 random-view blocks at stages 3,4,5 | M:73–75; U:296–319 |
| Basic blocks | Normalization and activation | GroupNorm; GELU. GN groups: largest divisor of C not exceeding 8. Framework epsilon/affine/activation approximation defaults: **Pending verification** | U:8–21,30–90 |
| Residual convolution | Kernels and strides | Two 3³ convolutions, padding 1; first stride 1 or 2, second stride 1; projected 1³ shortcut when required | U:110–144 |
| Depthwise separable convolution | Structure | DW 3³ with groups=C_in → GN/GELU → PW 1³ → GN/GELU; bias=False for these convolutions | U:59–90 |

### Random-view blocks, SE, and modality calibration

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| Random-view block | Active fraction and branch retention | Active channel fraction 0.5; branch retention probability 0.75 during training, with at least one retained branch; all four branches averaged in evaluation | U:172–179,243–268 |
| Random-view block | Branch kernels | (3,3,1), (3,1,3), (1,3,3), and dilated 3³ with dilation 2; depthwise convolution | U:181–231 |
| SE | Reduction | reduction 4; hidden=max(C_active/4 floored,8); GAP→1³→GELU→1³→sigmoid | U:93–107,239 |
| Modality affine | Learned calibration | Per-stage scale-increment and bias tensors: (4,64), (4,96), (4,128), initialized 0; effective scale 1+increment | M:96–98; U:271–280 |

### MAMF

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| MAMF | Scoring MLP | GAP→LayerNorm(C)→Linear(C,h)→GELU→Linear(h,1); h=16,16,32,48,64; separate modules across scales | M:100–104; U:328–348 |
| MAMF | Modality bias and masking | Learned bias (1,4,1), initialized 0; unavailable logits replaced by −10^4; softmax over modalities; **no explicit denominator epsilon** | U:335–348 |

### CSCA

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| CSCA | Scales and projections | Stages 3/4/5; 9 independent 1³ projections with GN/GELU to target widths 64/96/128; aligned features summed | U:352–383 |
| CSCA | Resizing and refinement | Trilinear, align_corners=False, for both up/down resizing; one residual refinement block with two DS combinations per target scale | U:24–27,364–383 |

### LPB and the prior head

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| LPB | Channels and grid | Input/bottleneck/output 128; adaptive pooling 4³, 64 spatial tokens/channel | F:10–12; U:413–442 |
| LPB | Low-rank projections | rank 16; W_down(16,64), W_up(64,16); bias=False, no inter-projection nonlinearity | U:389–406 |
| LPB | Additional refinements | Initial 1³/GN/GELU projection; two residual DS refinement blocks surrounding the token mixer; an additional pre-pool 1³/GN/GELU in the mixer | U:391,424–442 |
| LPB | Channel mixing and residual | Channel MLP 128→256→128; GN/GELU then GN; learned scalar initialized 0.5 for outer mixer residual | U:394–410 |
| LPB | State-space operation | None; no Mamba or state-space scan | M:9–31; U:386–442 |
| Prior head | Structure | Residual DS refinement 128→64 followed by 1³ projection 64→4; coarse softmax prior for decoder conditioning | U:445–453; M:223–224 |

### PGMSD and segmentation heads

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| PGMSD | Decoder widths | Four stages with output widths 96,64,32,16 | M:117–147 |
| PGMSD | Guide/skip/prior alignment | Trilinear resizing and 1³/GN/GELU projections; guide also has the preceding up_proj layer | U:468–471,530,542–545 |
| PGMSD | Gate MLP dimensions | 292→96→288; 196→64→192; 100→32→96; 52→32→48; GELU and final three-way sigmoid; no MLP normalization or Dropout | U:475–515 |
| PGMSD | Availability conditioning | Four-element availability mask concatenated with three GAP descriptors; input width 3C+4 | U:496–503 |
| PGMSD | Learned scalar initializations | Per-stage independent skip scale 1.0, prior scale 0.5, and outer residual scale 0.5 | U:483–484,539–546 |
| PGMSD | Feature refinement | One residual DS block inside the gate plus one post-refinement block; additional scaled projected-guide residual | U:482,516,539–546 |
| Final/auxiliary heads | Structure and placement | Residual DS refinement→1³ class projection; final at d1 (16 ch), auxiliary at d4/d3 (96/64 ch); logits resized to input size before softmax | M:150–152,274–280; U:549–558 |
| Native inference | Auxiliary execution | Native forward returns only final output but still computes auxiliary outputs when use_aux_heads=True; optimized runners must be specified separately | M:274–280 |

### Dropout and initialization

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| Dropout | Explicit Dropout layers | None in the listed full architecture; random-view branch retention is reported separately | M:34–152; U:30–558 |
| Framework initialization | Linear/Conv and norm defaults | No project-specific initialization for token Linear weights; exact training-framework defaults **Pending verification** | U:386–410; M:34–152 |

## Training protocol

The reference checkpoint records seed 999; the archived revision command uses seed 1024. Other runs must record their own effective configuration. Original-run settings must not be filled using the revision launch without supporting evidence.

| Parameter | Original manuscript run | Archived revision protocol / verified reference evaluation | Evidence |
|---|---|---|---|
| Optimizer | **Pending verification** of original args/state | AdamW constructor in the frozen revision entry | T:416 |
| Initial learning rate / weight decay | **Pending verification**; manuscript states 3e-4/3e-5 | Explicit revision launch 3e-4/3e-5; resume-state values require verification | I:34–37; T:425–432 |
| AdamW betas / epsilon / AMSGrad | **Pending verification** | **Pending verification** of runtime defaults or restored param_groups | T:416,428 |
| Training crop / nominal batch | **Pending verification** of original training args | 128³ / 2 in archived revision command; reference evaluation patch 128³ | I:28–43; F:28 |
| Total training epochs | **Pending verification** of original run linkage | 800 in revision launch; reference checkpoint archived at zero-based epoch 799 | I:26–27; F:6 |
| LR scheduler | **Pending verification** of original entry | Polynomial, power 0.9, updated at epoch start; no explicit min_lr floor or warmup in this entry | E:32–41; T:417,437–439 |
| Gradient clipping / accumulation | **Pending verification** for original run | No clipping call; one optimizer step per batch, accumulation factor 1 | T:463–478 |
| AMP enablement / dtype | **Pending verification** for original training | --amp enabled; exact autocast dtype **Pending verification**, not inferred as FP16/BF16 | I:67; T:418,465 |
| Activation checkpointing | **Pending verification** for original training | Disabled by revision launch; when enabled, the audited model uses use_reentrant=False | I:66; M:184–185,235–267 |
| TF32 / cuDNN benchmark | **Pending verification** for original training | Both enabled by revision launch; not a guarantee of deterministic execution | I:68–69; T:166–175 |
| Training seed | Archived reference checkpoint records 999; full original-run linkage **Pending verification** | This revision command uses 1024; other runs must list their own actual seeds | F:7; I:24–25 |
| DataLoader workers | **Pending verification** for original run | Train 4, validation/test 0; train shuffle=True, CUDA pinning enabled | I:30–31; T:227–249 |
| Validation/checkpoint selection | **Pending verification** of original log | Every 25 epochs and final epoch; best raw mean DSC over WT/TC/ET; last saved each epoch | I:32–33; T:547–615 |
| Pretraining/resumption | **Pending verification** | Initial archived revision launch has no --resume; later resume events require separate identification | H:17; T:423–433 |

## Loss functions

| Parameter | Original manuscript run | Archived revision protocol / verified reference evaluation | Evidence |
|---|---|---|---|
| Final/prior/auxiliary segmentation losses | Same implemented definition; original source binding **Pending verification** | Weighted CE + Dice, including background | L:7–58 |
| Loss coefficients | **Pending verification** of full original args; paper states 1/0.4/0.3/0.1 | Explicit revision command: final 1, prior 0.4, aux 0.3, consistency 0.1; auxiliary heads averaged | I:60–65; L:63–74 |
| Dice smoothing / CE clipping | Original implementation binding **Pending verification** | Denominator smooth 1e-7; logarithm uses probability clamp[0.005,1] | L:7–29 |
| Class weighting | Original implementation binding **Pending verification** | Patch-adaptive per-sample class-complement frequency, including background | L:20–29 |
| Prior alignment / consistency | Original binding and library default **Pending verification** | Prior logits resized to input size before softmax; F.mse_loss with default reduction; reduction **Pending verification** in original runtime | M:225–233; L:73 |

## Data processing and inference

| Parameter | Original manuscript run | Archived revision protocol / verified reference evaluation | Evidence |
|---|---|---|---|
| Augmentation | Original exact transform version **Pending verification** | Crop→rotation→intensity perturbation→three-axis flip; integer rotation−10…9°, order 0/cval−1; shift U(−.1,.1), scale U(.9,1.1); zero-padding added in revision crop | D:59–66; N0:33–144; N1:115–159 |
| Rotation-label boundary handling | Not triggered in the audited uint8-before-rotation pipeline; see verification scope | 1001/1001 BraTS2023 and 1000/1000 BraTS2021 label headers are uint8; 180 uint8 angle/plane checks and two actual-sample replays show no negative-index ET | [Verification evidence](#label-rotation-mechanism-and-verification) |
| Training modality availability | Original source binding **Pending verification** | Uniform sampling among 15 non-empty masks, probability 1/15 each, per training sample access | D:37–56,131–133 |
| Dataset partition | Original run binding / generation seed **Pending verification** | Explicit frozen BraTS2023 lists 1001/125/125; generation seed **Pending verification**, distinct from training seed | I:12–21; R0:13–27 |
| Sliding-window inference | Reference test configuration confirmed; original reported-result linkage **Pending verification** | 128³, overlap 50%, uniform probability averaging before argmax; no TTA | F:27–33; Q:251–310 |
| Inference precision | Original evaluation **Pending verification** | Archived segmentation re-evaluation FP32/TF32 on; PRISM-EFF-1.0 FP32/TF32 off | F:27–33; B:35–47,76–82 |
| DSC/HD95 | Original command-policy binding **Pending verification** | WT>0,TC{1,3},ET 3; double-empty DSC 1/HD 0; single-empty DSC 0/HD reference-volume diagonal; pooled bidirectional 26-neighbor surface 95th percentile,1 mm | Q:313–368; F:34–40 |

### Label-rotation verification and HD95 completeness

| Check | Verified result | Detailed evidence |
|---|---|---|
| Label dtype coverage | BraTS2023: 1001/1001 uint8; BraTS2021: 1000/1000 uint8 | [Dtype and data-flow evidence](#label-rotation-mechanism-and-verification) |
| Rotation mechanism | uint8 retained through rotation; conversion to int64 occurs afterward; tested boundary fill becomes background 0 | [Mechanism and scope](#label-rotation-mechanism-and-verification) |
| Synthetic checks | 180 uint8 angle/plane checks across three implementations; zero negative labels | Saved checks (`rotation_and_hd95_synthetic_checks.json`; archived evidence, not bundled) |
| Actual training-input replay | Two 128³ samples; negative labels 0/0; class-3 counts 7353→7353 and 12514→12514 through one-hot encoding | [Replay table](#actual-sample-replay) |
| Reference HD95 completeness | BraTS2023/BraTS2021: 5625/5625 region-level records each; non-finite HD95 0/0; single-empty records 104/81 retained | [Saved-record verification](#saved-record-verification) |

The label finding excludes the specific negative-fill-to-ET mechanism in the audited pipeline; it is not an exhaustive replay of all historical batches. Legacy HD95 protocols differ and require explicit harmonization before their values are compared with the current reference results.

## Label and HD95 verification

Consolidated from the supplied label-rotation and HD95 verification document (2026-11-02). These checks summarize existing audit evidence; no new training or inference was performed for this documentation merge. Read them together with [the public-code correspondence](#scope-and-public-code-correspondence).

### Label rotation: mechanism and verification

**Finding:** the negative-fill-to-ET mechanism is not triggered by the audited uint8 training-label pipeline. The earlier generic warning must be interpreted together with the actual label dtype and transform order.

The audited sequence is:

```text
load uint8 segmentation -> random crop -> rotate (uint8 output)
-> intensity transform leaves labels unchanged -> spatial flips
-> convert labels to int64 -> one-hot encoding
```

The conversion to int64 occurs after rotation. In the verified SciPy implementations, rotating uint8 labels with constant fill -1 produces background value 0 at the boundary, rather than a signed -1. Converting those non-negative labels to int64 does not introduce a negative index.

| Evidence level | Check | Result | Scope / evidence |
|---|---|---|---|
| Training-data coverage | Header scan of BraTS2023 training segmentations | 1001 / 1001 are uint8 | Existing reliability audit, Section 3.1; dtype scan, not full-volume augmentation of every case |
| Training-data coverage | Header scan of BraTS2021 training segmentations | 1000 / 1000 are uint8 | Existing reliability audit, Section 3.1 |
| Source data flow | Rotation precedes conversion to int64; label intensity transform is identity; flips preserve values | No signed conversion before rotation in this pipeline | Frozen dataset: lines 59–66,112–129; original transforms: lines 33–70,127–158 |
| Runtime behavior | uint8 rotation with negative constant fill | Boundary fill becomes 0; no negative output labels | Audit reports SciPy 1.15.3 locally and 1.17.1 on the server |
| Synthetic angle sweep | 3 implementations × 20 integer angles (-10 to 9 degrees) × 3 planes | 180 / 180 checks contain no negative labels for uint8 | Saved synthetic evidence; 16³ zero-label geometry checks, not a training contamination-rate estimate |
| Real-input replay | Two actual BraTS2023 training samples through the frozen data loader and 128³ augmentation | No negative labels; class-3 counts unchanged by one-hot encoding | Existing report, Section 3.1; replay results below |
| Positive control | Same synthetic chain with signed/float labels | Risk reproduces; four-class control: 8 class-3 voxels become 520 one-hot ET voxels, including 512 negative-fill voxels | Demonstrates that the check detects the failure when its dtype condition is present |

#### Actual-sample replay

| Replay sample | Label dtype before rotation | Angle / plane | Negative labels after rotation | Class-3 voxels after rotation | ET voxels after one-hot | Difference |
|---|---|---|---:|---:|---:|---:|
| A | uint8 | -5 degrees / (1,0) | 0 | 7353 | 7353 | 0 |
| B | uint8 | -9 degrees / (1,0) | 0 | 12514 | 12514 | 0 |

Sample identifiers are omitted from this public summary; the original audit retains the source case identifiers. Counts compare the rotated labels with their one-hot encoding, not with the unrotated segmentation, whose geometry may change legitimately.

**Mechanism-level interpretation.** Given the audited uint8 input, unchanged transform order, and verified runtime behavior, the specific path `negative rotation fill -> negative one-hot index -> false ET` cannot occur: no negative value reaches one-hot indexing. This conclusion is about that mechanism in this pipeline. The two real-sample replays validate execution; they are not an exhaustive replay of every historical batch. Signed/float inputs under the same augmentation code remain a conditional hazard.

### HD95: the protocol that must be stated explicitly

HD95 with empty masks uses an explicit evaluation convention. The volume-diagonal value below is a penalty for an otherwise undefined distance, not an observed surface distance.

| Prediction | Ground truth | DSC | HD95 handling in the archived reference evaluation |
|---|---|---|---|
| Empty | Empty | 1 | 0 mm |
| Empty | Non-empty | 0 | Volume-diagonal penalty; retain the record |
| Non-empty | Empty | 0 | Volume-diagonal penalty; retain the record |
| Non-empty | Non-empty | Case-wise Dice | 95th percentile of pooled bidirectional surface distances |

| HD95 parameter | Archived reference protocol |
|---|---|
| Spacing | (1,1,1) mm |
| Surface extraction | 26-neighbor erosion, 3×3×3 all-ones structure, border value 0; surface from XOR |
| Distance statistic | Pool both directed Euclidean surface-distance arrays, then take percentile 95 |
| Single-empty penalty | `sqrt(sum(((shape - 1) * spacing)^2))`, using the evaluated segmentation volume shape |
| Missing-value handling | No single-empty omission through NaN averaging |
| Non-empty-GT subset | Select using ground truth only, separately for WT/TC/ET; completely missed non-empty regions remain included |
| Aggregation | Case mean within each pattern and region; equal-weight mean over the 15 patterns when reporting All-15 |

#### Saved-record verification

The following numbers were read from existing patient-metric CSVs; no predictions were recomputed. A record is one patient × one pattern × one region, not a unique patient.

| Dataset / reference analysis | Unique patients | Patterns | Region-level records | Single-empty records | Double-empty records | Non-finite HD95 records | Completely missed non-empty-GT records retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| BraTS2023 | 125 | 15 | 5625 | 104 | 24 | 0 | 98 |
| BraTS2021 | 125 | 15 | 5625 | 81 | 9 | 0 | 30 |

Every single-empty record in these two CSVs has DSC=0 and finite positive HD95; every double-empty record has DSC=1 and HD95=0. The recorded single-empty counts by region are WT/TC/ET = 8/41/55 for BraTS2023 and 0/12/69 for BraTS2021.

| Dataset | Total patients | WT non-empty GT | TC non-empty GT | ET non-empty GT |
|---|---:|---:|---:|---:|
| BraTS2023 | 125 | 125 | 124 | 124 |
| BraTS2021 | 125 | 125 | 125 | 121 |

GT-nonempty filtering does not remove false-negative predictions. Report all-case and GT-nonempty results as a pair from the same evaluation; do not attach new GT-nonempty means to older all-case numbers obtained under another protocol.

### Why historical HD95 numbers cannot be relabeled

The existing audit establishes actual protocol differences, not merely alternative defaults:

| Record / implementation | Single-empty handling | Surface definition | Consequence |
|---|---|---|---|
| Historical main-test reports | NaN; actual nonzero NaN counts recorded | Must bind the individual historical implementation | Single-empty records can be omitted from means; this is not equivalent to the current penalty |
| Historical ablation implementation | `norm(shape * spacing)` | 6-neighbor surface | Both the penalty and finite-mask surface distances can differ from the current definition |
| Archived reference re-evaluation | `norm((shape - 1) * spacing)` | 26-neighbor surface | Explicit finite penalty and complete patient-level records |

An existing pure-metric CPU example with shape (4,5,6) gives 7.0710678 for the current penalty and 8.7749644 for the old penalty. Another stored example gives 26-neighbor HD95=1.6366996 and 6-neighbor HD95=1.7320508. These examples show non-equivalence; they are not estimates of the historical result changes.

### Resolution using existing outputs first

1. Use one explicitly specified protocol for every result being directly compared. The table above documents the archived reference protocol; this documentation update does not change any evaluation code.
2. Reuse complete patient-level outputs already obtained under that exact protocol. The two reference analyses above have complete, finite records.
3. For legacy outputs, if only NaN omission differs and identical surface-distance geometry is established, complete per-case records plus empty-case flags, shapes, and spacing can support correcting penalties and re-aggregation without inference.
4. If 6-neighbor and 26-neighbor geometry differ, scalar HD95 values cannot generally be converted. Recompute metrics from saved prediction masks and ground truth. This needs no model forward pass or retraining.
5. If only legacy summary means remain and prediction masks are unavailable, those summaries do not determine the corrected result. Inference would need separate scheduling; nothing was launched here.
6. Report all-case and GT-nonempty metrics together, preserve completely missed regions, disclose changed metric conventions, and regenerate the affected tables and textual comparisons from the unified outputs.

**Status:** uint8-pipeline evidence and the two reference-record checks are documented. Harmonization of every historical table is not claimed complete.

### Evidence files

- Stored synthetic CPU checks (`rotation_and_hd95_synthetic_checks.json`; archived evidence, not bundled): copied unchanged from the existing audit; includes positive controls and angle sweeps.
- Read-only verification summary (`label_and_hd95_verification_summary.json`; archived evidence, not bundled): aggregate CSV checks and source hashes; real-input scan/replay entries are explicitly attributed to the prior report.
- Source audit: `R2_PRISM_实验可靠性风险专项核查_20261102.md`, Sections 3.1 and 3.4. The original report contains internal source locations and is not copied into this public package.

Source audit SHA256: `514035c8d2c0f3382f4e346717878832ef8ed7f7ef14123739edd90e59b0e281`.

## Efficiency and software

| Parameter | Original manuscript run | Archived revision protocol / verified reference evaluation | Evidence |
|---|---|---|---|
| Estimated GFLOPs | Source of original reported value **Pending verification** | Hook-based partial operator count; 2 FLOPs/MAC; one Full 1×4×128³ patch for new benchmark; support set differs between old and new counters | Q:371–415; B:221–272 |
| Trainable parameters | Manuscript approximately 2.34 M; exact current recount **Pending verification** | requires_grad parameters only, including registered auxiliary heads; archived audit record 2,338,378 is secondary evidence | E:77–78; R:46 |
| Software versions | **Pending verification** for original training and evaluation | Re-evaluation manifest records Torch 2.11.0+cu128, not evidence of the original training version | F:179; requirements-server.txt:1–5 |

## Implementation notes and pending verification

<details>
<summary>Loss reduction and augmentation details</summary>

| Item | Implementation | Evidence |
|---|---|---|
| Class weights | Per sample and patch: `w[b,c] = 1 - sum_spatial(y[b,c]) / max(sum_class_and_spatial(y[b]), 1)`, including background. | L:20–29 |
| Weighted CE reduction | For each class, average `-w * y * log(clamp(p,0.005,1))` over batch/spatial axes; sum over classes. | L:24–29 |
| Dice reduction | Per-class sums over batch/spatial axes; average over four classes, including background. Denominator epsilon 1e-7; no numerator smoothing. | L:7–17 |
| Auxiliary averaging | Average the two auxiliary losses before applying 0.3; effective coefficient 0.15 per head in the archived revision protocol. | L:57–74; I:60–65 |
| Consistency gradients | Neither prior nor final probabilities are detached. Default MSE reduction in original runtime: **Pending verification**. | L:73 |
| Rotation sampling | Uniform plane selection: (1,0), (2,1), (2,0); integer angle -10 to 9 degrees. Both image/label: order 0, reshape=False, constant fill -1. | N0:33–70; D:63 |
| Flips | Each of three spatial axes independently flipped with probability 0.5. | N0:77–97 |
| Intensity perturbation | `x_new = x * scale + shift`; shift uniform [-0.1,0.1), scale uniform [0.9,1.1). Arrays (1,H,1,1,C) share values across the other two spatial axes. | N0:127–144; N1:142–159 |
| Padding | Revision training crop: symmetric zero padding before crop; older crop: no padding. Test windows: high-end padding, followed by cropping outputs to the original size. | N0:115–121; N1:115–136; Q:267–274,308–310 |
| Validation selection | Raw WT/TC/ET DSC mean; separately calculated ET<500-voxel postprocessing statistics are not used for best-checkpoint selection. | V:43–61; T:569–570 |

</details>

<details>
<summary>Configuration precedence and implementation discrepancies</summary>

- Reference model: base 16 / bottleneck 128. Constructor defaults: 24 / 192 (`M:34–53`).
- Frozen training parser imports only four Boolean preset defaults; numeric preset fields are not automatically applied (`T:113–148; P:26–42`).
- Non-full ablation presets can override individual architecture flags (`T:73–102; AB:45–61`).
- Resume restores optimizer groups and scaler; next scheduler update sets group-0 LR from current arguments (`T:425–439`).
- MAMF uses learned modality bias, masked logit -1e4, softmax; no explicit denominator epsilon (`U:335–348`).
- Weighted CE clamps probabilities before taking log (`L:24–29`).
- Decoder includes gate-internal refinement, a second post-refinement, and an additional projected-guide residual with a learned scale initially 0.5 (`U:482–516,539–546`).
- Native inference computes auxiliary predictions and full-resolution prior before returning only final output. Optimized runners must be identified separately (`M:223–233,274–280`).
- The signed-input rotation hazard does not trigger in the audited uint8 pipeline: dtype scans, source data flow, 180 uint8 synthetic angle/plane checks, and two actual-sample replays support this conclusion. See [the verification evidence](#label-rotation-mechanism-and-verification).
- Historical HD95 reports actually used NaN omission, and legacy ablations also used a different surface definition/diagonal. Current reference records use finite penalties with 26-neighbor surfaces. See [the protocol and resolution](#hd95-the-protocol-that-must-be-stated-explicitly); harmonization of historical tables is not claimed complete.

</details>

<details>
<summary>Profiling scope and remaining evidence</summary>

- Archived exact parameter count 2,338,378 is secondary evidence, not a new recount. Registered auxiliary parameters are included when requires_grad=True.
- Frozen FLOPs counter supports Conv3d/Linear; new counter also supports ConvTranspose3d. These are partial hook-based counts, 2 FLOPs/MAC; interpolation, normalization, activation, softmax, and other unhooked work are excluded (`Q:371–415; B:221–272`).
- New timing protocol: volume 240×240×155; 20 warmup / 50 timed passes; seed 20260727; FP32; TF32 off. These are code-specified settings, not new measurements (`B:35–47,161–218`).
- Original reported 134.31 GFLOPs still requires the original command, source snapshot, input, enabled heads, precision/backend settings, and raw output.
- Remaining evidence: original effective arguments, optimizer/scaler state, resume history, software environment, autocast dtype, library defaults, split-generation seed, and checkpoint-to-manuscript-result mapping.
- Prepared manifests or planned seeds do not establish run completion. Earlier 96³/200-epoch, 128³/300-epoch, and BraTS2021/300-epoch runs are separate protocols.

</details>

## Source evidence

The identities and SHA256 hashes below refer to archived audit snapshots. Those snapshots are not bundled here. Source keys and numeric lines throughout the tables refer to these identities, not to same-named public files. Current public-source links are provided separately in [Scope and public-code correspondence](#scope-and-public-code-correspondence).

<details>
<summary>Source-file identities and SHA256 hashes</summary>

| Key | Source identity | SHA256 |
|---|---|---|
| AB | Archived `ablation.py` (not bundled) | `230fa07168244d0af5956f375de782a15e6170d78bb80632380bcc0db8845812` |
| B | Archived `benchmark_efficiency.py` (not bundled) | `b2d68ecfebc3084e49753558dffa33ee0aff9a916469ee69821ec85bd4d9d020` |
| D | Archived `datasets.py` (D); public snapshot: **Pending verification** | `36f7ef36188cef3351a1bef789e6eaee51a678fb9a1c9270c487a9c3d52df176` |
| E | Archived `engine.py` (not bundled) | `dfe0d2566196a2661673babaea0e1fe276080510bc99359cc37e7302a0c182fa` |
| F | Archived `evaluation_manifest.json` (F); public snapshot: **Pending verification** | `fca9731f40aeb111b18eaf5d4eb1954602d74609837c212a9155d7f9a5602772` |
| H | Archived `run_seed_1024.sh` (H); public snapshot: **Pending verification** | `14b02887a8db5dde26b58f0c860e88d7595a27e2bdeee36241b8df0a2e9aac23` |
| I | Archived `main_reference_command.json` (I); public snapshot: **Pending verification** | `f1bb38059f60a8e2a9a503924bbfab9aa5638c4fddaadb44b864b529291a493a` |
| L | Archived `losses.py` (not bundled) | `489e7b0777b5354624a167736fd2a5a08069b1c9c63913e7497ddd94689d6173` |
| M | Archived `model.py` (not bundled) | `78280bae799bd8b113849dad1271568ecf78336b5790ff4ed44703ef07a80ebe` |
| N0 | Archived `transforms.py` (N0); public snapshot: **Pending verification** | `706a695ae98b0f7b2de24b765ff6242b24718056d39750044ffeb66dbbf3cef7` |
| N1 | Archived `transforms.py` (N1); public snapshot: **Pending verification** | `61746272820e188049280e835c06718a1b31ee309fc51a0fa9eeb6b77840118d` |
| P | Archived `presets.py` (not bundled) | `f8ddf66f507997c03a8da629966495d99870cee395464398d022254b7d1e5317` |
| Q | Archived `test.py` (Q); public snapshot: **Pending verification** | `94fc5f34b6959c2e0fe74be3a2dde63864d07a23f1b90f232c76609edbf1eb9a` |
| R | Archived `服务器核查与启动命令_2026-10-26.md` (R); public snapshot: **Pending verification** | `b70ba87eb61bda2b8b69a2c561cf357199c55f77701c16e986a249f6353063b9` |
| R0 | Archived `input_manifest.json` (R0); public snapshot: **Pending verification** | `eeaf14e4a795b36d36a0295511ed6658c7bf4148db3b648b0a4ca1f8c87f01ab` |
| T | Archived `train_poly.py` (T); public snapshot: **Pending verification** | `4da72cccba7b2a785b567da70d23dc8ad792e796beea687670389060975ced2c` |
| U | Archived `modules.py` (not bundled) | `d0198b99627470da4f709d555394ae1ac2dc4c7fdf9fa74f5e1f11cda3ac5759` |
| V | Archived `evaluate.py` (not bundled) | `41a70d3cfd1094e8f72883f23fff02c31e1fbca1d78540078e8fd6f33e187873` |

Evidence uses numeric source lines; the source identity and hash distinguish snapshots with the same filename.

</details>
