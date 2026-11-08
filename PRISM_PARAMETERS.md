# PRISM: Parameters and Reproducibility Settings

Configuration reference for *PRISM: Lightweight Prior-Guided Segmentation of Incomplete Multimodal Brain Tumor MRI*.

**Documentation snapshot:** 2026-11-08, current-partition BraTS2023 re-audit and reference evaluation. **Status:** verified label coverage, augmentation replays and complete 249-case metrics.

This document lists the reference architecture, data-processing settings, inference details, and label/HD95 verification tables.

## Contents

- [Architecture](#architecture)
- [Data processing and inference](#data-processing-and-inference)
- [Label and HD95 verification](#label-and-hd95-verification)
- [Implementation notes](#implementation-notes)
- [Source evidence](#source-evidence)

**Reference configuration:** encoder widths (16,32,64,96,128); decoder widths (96,64,32,16); bottleneck 128 channels, pooled grid 4³, rank 16; approximately 2.34M trainable parameters.

**Notation:** C: channel width; DW/PW: depthwise/pointwise convolution; DS: depthwise separable convolution; GN: GroupNorm; GAP: global average pooling. A kernel written as 3³ means 3 × 3 × 3. Keys such as `U:322–349` identify source-file lines in [Source evidence](#source-evidence).

## Architecture

### Inputs, PSE, and basic blocks

| Component | Parameter | Setting | Evidence |
|---|---|---|---|
| Input/output | Modalities and classes | 4 modalities in the order FLAIR, T1ce, T1, T2; 4 segmentation classes including background | D:84–94; F:8–23; M:37–38 |
| PSE | Stage widths | 16,32,64,96,128 | F:8–13; M:49–53 |
| PSE | Modality-specific stems | Four independent stems; residual Conv3D 1→16, then stride-2 residual Conv3D 16→32 and a residual depthwise-separable refinement block | M:68–69; U:283–293 |
| PSE | Shared deep stages | Stride-2 residual blocks 32→64→96→128; 1,1,2 random-view blocks at stages 3,4,5 | M:73–75; U:296–319 |
| Basic blocks | Normalization and activation | GroupNorm; GELU. GN groups: largest divisor of C not exceeding 8. | U:8–21,30–90 |
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
| MAMF | Modality bias and masking | Learned bias (1,4,1), initialized 0; unavailable logits replaced by −10^4; softmax over modalities | U:335–348 |

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

## Data processing and inference

| Parameter | Original manuscript run | Verified reference evaluation | Evidence |
|---|---|---|---|
| Rotation-label boundary handling | Not triggered in the audited uint8-before-rotation pipeline; see verification scope | 875/875 BraTS2023 and 1000/1000 BraTS2021 label headers are uint8; 180 uint8 angle/plane checks and two actual-sample replays show no negative-index ET | [Verification evidence](#label-rotation-mechanism-and-verification) |

### Label-rotation verification and HD95 completeness

| Check | Verified result | Detailed evidence |
|---|---|---|
| Label dtype coverage | BraTS2023: 875/875 uint8; BraTS2021: 1000/1000 uint8 | [Dtype and data-flow evidence](#label-rotation-mechanism-and-verification) |
| Rotation mechanism | uint8 retained through rotation; conversion to int64 occurs afterward; tested boundary fill becomes background 0 | [Mechanism and scope](#label-rotation-mechanism-and-verification) |
| Synthetic checks | 180 uint8 angle/plane checks across three implementations; zero negative labels | Saved checks (`rotation_and_hd95_synthetic_checks.json`; archived evidence, not bundled) |
| Actual training-input replay | Two samples from the current 875-scan training partition, cropped to 128³; negative labels 0/0; class-3 counts 22600→22600 and 24256→24256 through one-hot encoding | [Replay table](#actual-sample-replay) |
| Reference HD95 completeness | BraTS2023: 11205 region-level records on 249 cases, 0 non-finite HD95 records and 196 single-empty records retained. BraTS2021: 5625 records on 125 cases, 0 non-finite and 81 single-empty records retained. | [Saved-record verification](#saved-record-verification) |

The label finding excludes the specific negative-fill-to-ET mechanism in the audited pipeline; it is not an exhaustive replay of all historical batches. Legacy HD95 protocols differ and require explicit harmonization before their values are compared with the current reference results.

## Label and HD95 verification

BraTS2023 label coverage and augmentation replays were checked against the current 875/125/249 patient-disjoint partition. DSC, HD95 and empty-mask record counts below were evaluated on all 249 test scans using the matching seed-999 checkpoint. BraTS2021 retains its existing 125-case reference evaluation.

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
| Training-data coverage | Header scan of BraTS2023 training segmentations | 875 / 875 are uint8 | New header scan over all 875 scans in the current training partition |
| Training-data coverage | Header scan of BraTS2021 training segmentations | 1000 / 1000 are uint8 | Existing reliability audit, Section 3.1 |
| Source data flow | Rotation precedes conversion to int64; label intensity transform is identity; flips preserve values | No signed conversion before rotation in this pipeline | Frozen dataset: lines 59–66,112–129; original transforms: lines 33–70,127–158 |
| Runtime behavior | uint8 rotation with negative constant fill | Boundary fill becomes 0; no negative output labels | Current-partition replay uses SciPy 1.17.1 |
| Synthetic angle sweep | 3 implementations × 20 integer angles (-10 to 9 degrees) × 3 planes | 180 / 180 checks contain no negative labels for uint8 | Saved synthetic evidence; 16³ zero-label geometry checks, not a training contamination-rate estimate |
| Real-input replay | Two samples from the current BraTS2023 training partition through the audited 128³ augmentation chain | No negative labels; class-3 counts unchanged by one-hot encoding | Current-partition replay results below |
| Positive control | Same synthetic chain with signed/float labels | Risk reproduces; four-class control: 8 class-3 voxels become 520 one-hot ET voxels, including 512 negative-fill voxels | Demonstrates that the check detects the failure when its dtype condition is present |

#### Actual-sample replay

| Replay sample | Label dtype before rotation | Angle / plane | Negative labels after rotation | Class-3 voxels after rotation | ET voxels after one-hot | Difference |
|---|---|---|---:|---:|---:|---:|
| A | uint8 | -5 degrees / (1,0) | 0 | 22600 | 22600 | 0 |
| B | uint8 | -9 degrees / (1,0) | 0 | 24256 | 24256 | 0 |

Samples A and B are the first two eligible entries in the sorted current 875-scan BraTS2023 training list, requiring spatial axes of at least 128 voxels and non-empty ET. Python and NumPy random seeds are 999 for replay A and 1000 for replay B. Counts compare the rotated labels with their one-hot encoding, not with the unrotated segmentation, whose geometry may change legitimately.

**Mechanism-level interpretation.** Given the audited uint8 input, unchanged transform order, and verified runtime behavior, the specific path `negative rotation fill -> negative one-hot index -> false ET` cannot occur: no negative value reaches one-hot indexing. This conclusion is about that mechanism in this pipeline. The two real-sample replays validate execution; they are not an exhaustive replay of every historical batch. Signed/float inputs under the same augmentation code remain a conditional hazard.

### HD95: the protocol that must be stated explicitly

HD95 with empty masks uses an explicit evaluation convention. The volume-diagonal value below is a penalty for an otherwise undefined distance, not an observed surface distance.

| Prediction | Ground truth | DSC | HD95 handling in the reference evaluation |
|---|---|---|---|
| Empty | Empty | 1 | 0 mm |
| Empty | Non-empty | 0 | Volume-diagonal penalty; retain the record |
| Non-empty | Empty | 0 | Volume-diagonal penalty; retain the record |
| Non-empty | Non-empty | Case-wise Dice | 95th percentile of pooled bidirectional surface distances |

| HD95 parameter | Reference protocol |
|---|---|
| Spacing | (1,1,1) mm |
| Surface extraction | 26-neighbor erosion, 3×3×3 all-ones structure, border value 0; surface from XOR |
| Distance statistic | Pool both directed Euclidean surface-distance arrays, then take percentile 95 |
| Single-empty penalty | `sqrt(sum(((shape - 1) * spacing)^2))`, using the evaluated segmentation volume shape |
| Missing-value handling | No single-empty omission through NaN averaging |
| Non-empty-GT subset | Select using ground truth only, separately for WT/TC/ET; completely missed non-empty regions remain included |
| Aggregation | Case mean within each pattern and region; equal-weight mean over the 15 patterns when reporting All-15 |

#### Current BraTS2023 partition and ground-truth region counts

The BraTS2023 patient-disjoint partition was audited on 2026-11-07. Patient identifiers group all time points of the same patient. Training, validation and test sets have no overlapping scan identifiers or patient identifiers.

| Partition | Scans | Unique patients |
|---|---:|---:|
| Training | 875 | 757 |
| Validation | 125 | 125 |
| Testing | 249 | 249 |
| Total | 1249 | 1131 |

The current BraTS2023 counts below were obtained by reading all 249 test segmentation masks. WT contains labels greater than 0, TC contains labels 1 or 3, and ET contains label 3. ET is empty in 12 cases (4.82%).

| Dataset / cohort | Total patients | WT non-empty GT | TC non-empty GT | ET non-empty GT |
|---|---:|---:|---:|---:|
| BraTS2023, current patient-disjoint test cohort | 249 | 249 | 249 | 237 |
| BraTS2021, archived evaluation cohort | 125 | 125 | 125 | 121 |

The BraTS2023 ground-truth counts and metric results below use the same 249-case test cohort.

#### All-case and GT-nonempty results

BraTS2023 uses the seed-999 checkpoint whose recorded training, validation and test lists match the current 875/125/249 partition; the test patients have no overlap with the training or validation patients. BraTS2021 uses its existing seed-999 reference evaluation on 125 cases. All-case and GT-nonempty columns use the same predictions within each dataset and the 26-neighbor evaluation protocol above.

The BraTS2023 evaluation uses 128³ sliding windows with 50% overlap and uniform averaging of probabilities before argmax. Computation uses FP32 with TF32 enabled, without test-time augmentation or small-ET postprocessing.

For each region, the GT-nonempty subset contains patients with at least one ground-truth voxel in that region. A missed region with non-empty ground truth retains its DSC of 0 and the volume-diagonal HD95 penalty. Each score averages patients within a modality pattern and then gives equal weight to all 15 patterns. Counts are unique patients within each dataset and region.

| Dataset | Region | GT-nonempty patients / total | All-case DSC (%) | GT-nonempty DSC (%) | All-case HD95 (mm) | GT-nonempty HD95 (mm) |
|---|---|---:|---:|---:|---:|---:|
| BraTS2023 | WT | 249 / 249 | 90.58 | 90.58 | 4.58 | 4.58 |
| BraTS2023 | TC | 249 / 249 | 85.14 | 85.14 | 7.71 | 7.71 |
| BraTS2023 | ET | 237 / 249 | 75.72 | 77.05 | 13.28 | 7.48 |
| BraTS2021 | WT | 125 / 125 | 89.65 | 89.65 | 5.51 | 5.51 |
| BraTS2021 | TC | 125 / 125 | 83.49 | 83.49 | 6.73 | 6.73 |
| BraTS2021 | ET | 121 / 125 | 71.93 | 73.81 | 13.81 | 7.24 |

#### Saved-record verification

BraTS2023 counts were recomputed from the completed 249-case evaluation. BraTS2021 counts come from the existing 125-case reference records. One record corresponds to one patient, one availability pattern and one tumor region.

| Dataset / reference evaluation | Unique patients | Patterns | Region-level records | Single-empty records | Double-empty records | Non-finite HD95 records | Completely missed non-empty-GT records retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| BraTS2023 | 249 | 15 | 11205 | 196 | 89 | 0 | 105 |
| BraTS2021 | 125 | 15 | 5625 | 81 | 9 | 0 | 30 |

Every single-empty record in these two CSVs has DSC=0 and finite positive HD95; every double-empty record has DSC=1 and HD95=0. The recorded single-empty counts by region are WT/TC/ET = 3/48/145 for BraTS2023 and 0/12/69 for BraTS2021.

GT-nonempty filtering does not remove false-negative predictions. Report all-case and GT-nonempty results as a pair from the same evaluation; do not attach new GT-nonempty means to older all-case numbers obtained under another protocol.

### Evidence files

- Current BraTS2023 audit covers all 875 training-label headers and all 249 test ground-truth masks. Augmentation replays use two samples from the current training partition.
- The new evaluation contains 11205 unique case-pattern-region records. Independent checks verified complete coverage, finite metrics, empty-mask rules and the published aggregate means. Encoder-cache validation passed 30 real patch-pattern comparisons with bitwise-identical probabilities.
- Training-list SHA256: `1bf95a2c6c9983bac79e428df0c817a6c3cf8b19f6d13ca13d568ea4f6c2ee68`.
- Test-list SHA256: `e8f6a7d646ac5caa6a1160259774297a42fce6b010bff6afe4679d53474dc7b6`.
- New patient-metric CSV SHA256: `c89e06ab5f40d0047a71fd48d2b5f0fcc93f6704d719bcc30ae6fde97f78c270`.
- New evaluation-manifest SHA256: `a961ac442bfe0c228c724dd17a0b1b432c5a1fad27c0137dbab04ede3d7aa4bf`.
- BraTS2023 checkpoint SHA256: `17728fdb9f16ac32fecedd5ae6185056702c51ca74cf3ce293182165d423bbbc`.

- Stored synthetic CPU checks (`rotation_and_hd95_synthetic_checks.json`; archived evidence, not bundled): copied unchanged from the existing audit; includes positive controls and angle sweeps.
- Read-only verification summary (`label_and_hd95_verification_summary.json`; archived evidence, not bundled): aggregate CSV checks and source hashes; real-input scan/replay entries are explicitly attributed to the prior report.
- Source audit: `R2_PRISM_实验可靠性风险专项核查_20261102.md`, Sections 3.1 and 3.4. The original report contains internal source locations and is not copied into this public package.

Source audit SHA256: `514035c8d2c0f3382f4e346717878832ef8ed7f7ef14123739edd90e59b0e281`.

## Implementation notes

<details>
<summary>Augmentation details</summary>

| Item | Implementation | Evidence |
|---|---|---|
| Rotation sampling | Uniform plane selection: (1,0), (2,1), (2,0); integer angle -10 to 9 degrees. Both image/label: order 0, reshape=False, constant fill -1. | N0:33–70; D:63 |
| Flips | Each of three spatial axes independently flipped with probability 0.5. | N0:77–97 |
| Intensity perturbation | `x_new = x * scale + shift`; shift uniform [-0.1,0.1), scale uniform [0.9,1.1). Arrays (1,H,1,1,C) share values across the other two spatial axes. | N0:127–144; N1:142–159 |
| Padding | Revision training crop: symmetric zero padding before crop; older crop: no padding. Test windows: high-end padding, followed by cropping outputs to the original size. | N0:115–121; N1:115–136; Q:267–274,308–310 |

</details>

<details>
<summary>Configuration precedence and implementation discrepancies</summary>

- Reference model: base 16 / bottleneck 128. Constructor defaults: 24 / 192 (`M:34–53`).
- MAMF uses learned modality bias, masked logit -1e4, softmax (`U:335–348`).
- Decoder includes gate-internal refinement, a second post-refinement, and an additional projected-guide residual with a learned scale initially 0.5 (`U:482–516,539–546`).
- Native inference computes auxiliary predictions and full-resolution prior before returning only final output. Optimized runners must be identified separately (`M:223–233,274–280`).
- The signed-input rotation hazard does not trigger in the audited uint8 pipeline: dtype scans, source data flow, 180 uint8 synthetic angle/plane checks, and two actual-sample replays support this conclusion. See [the verification evidence](#label-rotation-mechanism-and-verification).
- The reference HD95 protocol uses finite penalties with 26-neighbor surfaces. See [the metric definition](#hd95-the-protocol-that-must-be-stated-explicitly).

</details>

<details>
<summary>Profiling scope and remaining evidence</summary>

- Archived exact parameter count 2,338,378 is secondary evidence, not a new recount. Registered auxiliary parameters are included when requires_grad=True.
- Frozen FLOPs counter supports Conv3d/Linear; new counter also supports ConvTranspose3d. These are partial hook-based counts, 2 FLOPs/MAC; interpolation, normalization, activation, softmax, and other unhooked work are excluded (`Q:371–415; B:221–272`).
- New timing protocol: volume 240×240×155; 20 warmup / 50 timed passes; seed 20260727; FP32; TF32 off. These are code-specified settings, not new measurements (`B:35–47,161–218`).
- Original reported 134.31 GFLOPs still requires the original command, source snapshot, input, enabled heads, precision/backend settings, and raw output.

</details>

## Source evidence

The identities and SHA256 hashes below refer to archived audit snapshots. Those snapshots are not bundled here. Source keys and numeric lines throughout the tables refer to these identities, not to same-named public files.

<details>
<summary>Source-file identities and SHA256 hashes</summary>

| Key | Source identity | SHA256 |
|---|---|---|
| AB | Archived `ablation.py` (not bundled) | `230fa07168244d0af5956f375de782a15e6170d78bb80632380bcc0db8845812` |
| B | Archived `benchmark_efficiency.py` (not bundled) | `b2d68ecfebc3084e49753558dffa33ee0aff9a916469ee69821ec85bd4d9d020` |
| D | Archived `datasets.py` (D) | `36f7ef36188cef3351a1bef789e6eaee51a678fb9a1c9270c487a9c3d52df176` |
| E | Archived `engine.py` (not bundled) | `dfe0d2566196a2661673babaea0e1fe276080510bc99359cc37e7302a0c182fa` |
| F | Archived `evaluation_manifest.json` (F) | `fca9731f40aeb111b18eaf5d4eb1954602d74609837c212a9155d7f9a5602772` |
| H | Archived `run_seed_1024.sh` (H) | `14b02887a8db5dde26b58f0c860e88d7595a27e2bdeee36241b8df0a2e9aac23` |
| I | Archived `main_reference_command.json` (I) | `f1bb38059f60a8e2a9a503924bbfab9aa5638c4fddaadb44b864b529291a493a` |
| L | Archived `losses.py` (not bundled) | `489e7b0777b5354624a167736fd2a5a08069b1c9c63913e7497ddd94689d6173` |
| M | Archived `model.py` (not bundled) | `78280bae799bd8b113849dad1271568ecf78336b5790ff4ed44703ef07a80ebe` |
| N0 | Archived `transforms.py` (N0) | `706a695ae98b0f7b2de24b765ff6242b24718056d39750044ffeb66dbbf3cef7` |
| N1 | Archived `transforms.py` (N1) | `61746272820e188049280e835c06718a1b31ee309fc51a0fa9eeb6b77840118d` |
| P | Archived `presets.py` (not bundled) | `f8ddf66f507997c03a8da629966495d99870cee395464398d022254b7d1e5317` |
| Q | Archived `test.py` (Q) | `94fc5f34b6959c2e0fe74be3a2dde63864d07a23f1b90f232c76609edbf1eb9a` |
| R | Archived `服务器核查与启动命令_2026-10-26.md` (R) | `b70ba87eb61bda2b8b69a2c561cf357199c55f77701c16e986a249f6353063b9` |
| R0 | Archived `input_manifest.json` (R0) | `eeaf14e4a795b36d36a0295511ed6658c7bf4148db3b648b0a4ca1f8c87f01ab` |
| T | Archived `train_poly.py` (T) | `4da72cccba7b2a785b567da70d23dc8ad792e796beea687670389060975ced2c` |
| U | Archived `modules.py` (not bundled) | `d0198b99627470da4f709d555394ae1ac2dc4c7fdf9fa74f5e1f11cda3ac5759` |
| V | Archived `evaluate.py` (not bundled) | `41a70d3cfd1094e8f72883f23fff02c31e1fbca1d78540078e8fd6f33e187873` |

Evidence uses numeric source lines; the source identity and hash distinguish snapshots with the same filename.

</details>
