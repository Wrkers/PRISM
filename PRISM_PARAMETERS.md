# PRISM: Parameters and Reproducibility Settings

Configuration reference for *PRISM: Lightweight Prior-Guided Segmentation of Incomplete Multimodal Brain Tumor MRI*.

**Documentation snapshot:** 2026-11-08. BraTS2023 data, label checks and empty-mask record verification use the 875/125/249 patient-disjoint partition. Metric-table entries retain their manuscript source.

This document lists the reference architecture, data-processing settings, inference details, and label/HD95 verification tables.

## Contents

- [Architecture](#architecture)
- [Dataset Partition Details](#dataset-partition-details)
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
| Input/output | Modalities and classes | 4 modalities in the order FLAIR, T1ce, T1, T2; 4 segmentation classes including background | D:84–94; M:37–38 |
| PSE | Stage widths | 16,32,64,96,128 | M:49–53 |
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
| LPB | Channels and grid | Input/bottleneck/output 128; adaptive pooling 4³, 64 spatial tokens/channel | U:413–442 |
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

## Dataset Partition Details

BraTS2023 is used for the main comparative experiments, whereas BraTS2021 is used for ablation studies. Both datasets use fixed, non-overlapping patient-level partitions throughout their respective experiments. The **split seed** controls data partitioning and should not be confused with a **model-training or checkpoint seed** (e.g., 999 for an audited BraTS2023 checkpoint).

**BraTS2023.** We follow the patient-disjoint partition associated with IM-Fuse (upstream split seed **42**). All longitudinal scans of the same patient are grouped into one subset: 118 patients with multiple time points contribute 236 scans, all assigned to training. The other 1,015 patients initially contribute one scan each. After two scans were excluded from the test list, the evaluated cohort contains **1,249 scans from 1,131 patients**. Because the upstream partition script enumerates directories without sorting, **the frozen split lists used in the experiments, rather than seed 42 alone, define exact membership**. All main comparative methods were evaluated using this fixed partition, as reported in the manuscript.

**BraTS2021.** The 1,251 initial scan identifiers were sorted lexicographically and shuffled with `random.Random(999)`, then assigned to training/validation/testing as **1,001/125/125**. One training scan was excluded because its preprocessed depth was insufficient for the original crop configuration. The final partition is **1,000/125/125**, with one unique patient identifier per retained scan.

### Final split sizes and patient-level audit

| Dataset | Subset | Scans | Unique patients | Share of retained scans |
|---|---|---:|---:|---:|
| BraTS2023 | Training | 875 | 757 | 70.06% |
| BraTS2023 | Validation | 125 | 125 | 10.01% |
| BraTS2023 | Testing | 249 | 249 | 19.94% |
| **BraTS2023** | **Total** | **1,249** | **1,131** | **100%** |
| BraTS2021 | Training | 1,000 | 1,000 | 80.00% |
| BraTS2021 | Validation | 125 | 125 | 10.00% |
| BraTS2021 | Testing | 125 | 125 | 10.00% |
| **BraTS2021** | **Total** | **1,250** | **1,250** | **100%** |

| Patient-level verification item | BraTS2023 | BraTS2021 |
|---|---|---|
| Unit of separation | Patient ID, grouping multiple time points | Unique patient ID for each scan |
| Repeated scans | 118 patients / 236 scans, all in training | None in the retained cohort |
| Patient IDs shared between train/validation/test | **0** (pairwise intersections) | **0** (pairwise intersections) |
| Exclusions after initial allocation | 2 test scans | 1 training scan |
| Split-generation seed | 42 (upstream script; frozen lists define membership) | 999 (`random.Random`, after sorted identifiers) |

**Reproducibility boundary.** The repository's preprocessing code describes how the partitions are generated; this document does **not** publish patient-ID lists. For BraTS2023, matching seed 42 without matching the upstream directory iteration order does not guarantee identical membership. The SHA256 values of the frozen training, validation and test lists are provided in [Evidence files](#evidence-files) for checking a local copy against the experiment's lists; a hash alone cannot reconstruct a list. The audit figures above describe the recorded partitions, not an independent rerun performed from this Markdown document. For the 249-case BraTS2023 test-set audit and related evaluation evidence, see [Current BraTS2023 partition and ground-truth region counts](#current-brats2023-partition-and-ground-truth-region-counts).

## Data processing and inference

| Parameter | Pipeline finding | Current verification | Evidence |
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

The label finding concerns the specific negative-fill-to-ET mechanism in the audited pipeline. The real-input replays validate two training samples; the empty-mask record audit covers all 249 BraTS2023 test scans and all 15 availability patterns.

## Label and HD95 verification

BraTS2023 uses a single patient-disjoint partition of 875 training, 125 validation and 249 test scans. Label coverage, augmentation replays, ground-truth counts and the prediction-dependent empty-mask record audit use this partition. The record audit uses a seed-999 checkpoint whose training, validation and test lists match these lists, with no test-patient overlap with training or validation. BraTS2021 retains its 125-case reference evaluation.

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
| Single-empty penalty | `sqrt(sum(((shape - 1) * spacing)^2))`, calculated separately for each case from its entire preprocessed cropped segmentation array |
| Missing-value handling | No single-empty omission through NaN averaging |
| Non-empty-GT subset | Select using ground truth only, separately for WT/TC/ET; completely missed non-empty regions remain included |
| Aggregation | Case mean within each pattern and region; equal-weight mean over the 15 patterns when reporting All-15 |

#### BraTS2023 evaluation grid and case-specific penalties

For case i, `shape = (H_i,W_i,D_i)` is the spatial shape of its saved, preprocessed segmentation array. The physical penalty is `P_i = sqrt(((H_i-1)*s_x)^2 + ((W_i-1)*s_y)^2 + ((D_i-1)*s_z)^2)`, with `(s_x,s_y,s_z) = (1,1,1)` mm. The three axes span 128–156, 149–191 and 128–153 voxels across the 249 test scans. These are per-axis ranges; each case retains its own shape.

| Quantity | Shape / value | Role |
|---|---|---|
| Evaluated segmentation grid | Per-case preprocessed cropped array; 245 distinct shapes across 249 scans | Determines each case's single-empty penalty |
| Case-wise volume diagonals | 237.13–280.18 mm across the 249 scans | Finite, case-specific penalties |
| Example evaluated grid | 150×167×139 voxels; diagonal 262.30 mm | One actual ET-empty test case |
| Sliding window | 128×128×128 voxels, 50% overlap | Prediction probabilities are combined over the entire evaluated grid |
| Timing input | 240×240×155 voxels | Fixed input for the separate timing protocol described under profiling |

All 196 single-empty records were checked against their case-specific evaluated-grid diagonals. The optimized surface-distance routine handles empty masks on the entire evaluated grid before applying its local bounding-box optimization to double-nonempty masks. For ET, the 91 GT-empty/prediction-nonempty records have a mean penalty of 253.017124 mm; the 54 GT-nonempty/prediction-empty records have a mean penalty of 255.777422 mm. Their combined 145-record mean is 254.045097 mm. These are record-weighted summaries of case-specific values.

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

The BraTS2023 ground-truth counts and the saved-record audit below refer to the 249 test scans in the 875/125/249 partition. The metric entries in the following table retain their manuscript source.

#### All-case and GT-nonempty results

The DSC and HD95 entries in this table are retained from the manuscript at their existing precision. The manuscript identifies the BraTS2023 data partition by seed 42; BraTS2021 uses its seed-999 reference evaluation. Record-count and penalty evidence in the following audit sections is scoped to the independently identified prediction CSV and evaluation manifest.

The 249-case BraTS2023 record audit uses 128³ sliding windows with 50% overlap and uniform averaging of probabilities before argmax. Computation uses FP32 with TF32 enabled, without test-time augmentation or small-ET postprocessing.

For each region, the GT-nonempty subset contains patients with at least one ground-truth voxel in that region. A missed region with non-empty ground truth retains its DSC of 0 and the volume-diagonal HD95 penalty. Each score averages patients within a modality pattern and then gives equal weight to all 15 patterns. Counts are unique patients within each dataset and region.

| Dataset | Region | GT-nonempty patients / total | All-case DSC (%) | GT-nonempty DSC (%) | All-case HD95 (mm) | GT-nonempty HD95 (mm) |
|---|---|---:|---:|---:|---:|---:|
| BraTS2023 | WT | 249 / 249 | 90.0 | 90.0 | 5.9 | 5.9 |
| BraTS2023 | TC | 249 / 249 | 83.1 | 83.1 | 7.6 | 7.6 |
| BraTS2023 | ET | 237 / 249 | 75.1 | 75.6 | 13.3 | 7.5 |
| BraTS2021 | WT | 125 / 125 | 89.7 | 89.7 | 5.5 | 5.5 |
| BraTS2021 | TC | 125 / 125 | 83.5 | 83.5 | 6.7 | 6.7 |
| BraTS2021 | ET | 121 / 125 | 71.9 | 73.8 | 13.8 | 7.2 |

#### Saved-record verification

BraTS2023 counts were recomputed from the completed 249-case evaluation. BraTS2021 counts come from the existing 125-case reference records. One record corresponds to one patient, one availability pattern and one tumor region.

| Dataset / reference evaluation | Unique patients | Patterns | Region-level records | Single-empty records | Double-empty records | Non-finite HD95 records | Completely missed non-empty-GT records retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| BraTS2023 | 249 | 15 | 11205 | 196 | 89 | 0 | 105 |
| BraTS2021 | 125 | 15 | 5625 | 81 | 9 | 0 | 30 |

Every single-empty record in these two CSVs has DSC=0 and finite positive HD95; every double-empty record has DSC=1 and HD95=0. The recorded single-empty counts by region are WT/TC/ET = 3/48/145 for BraTS2023 and 0/12/69 for BraTS2021.

The BraTS2023 empty-mask counts can also be separated as follows. GT-empty and prediction-empty counts each include double-empty records. Single-empty means that exactly one of prediction and GT is empty; completely missed non-empty GT means GT is non-empty and prediction is empty.

| BraTS2023 region | GT-empty records | Prediction-empty records | Double-empty records | Single-empty records | Completely missed non-empty-GT records |
|---|---:|---:|---:|---:|---:|
| WT | 0 | 3 | 0 | 3 | 3 |
| TC | 0 | 48 | 0 | 48 | 48 |
| ET | 180 | 143 | 89 | 145 | 54 |
| Total | 180 | 194 | 89 | 196 | 105 |

These are case-pattern-region records. The 12 ET-empty test patients contribute 12×15 = 180 GT-empty ET records. The 89 double-empty ET records and 91 GT-empty/prediction-nonempty ET records together account for those 180 records.

GT-nonempty filtering retains false-negative predictions. All-case and GT-nonempty results must be derived from the same case list, predictions and evaluation protocol. Dataset-dependent summaries, including individual availability patterns, missing-modality groups and any uncertainty or significance estimates, require the same fixed partition.

### Evidence files

- Current BraTS2023 audit covers all 875 training-label headers and all 249 test ground-truth masks. Augmentation replays use two samples from the current training partition.
- The audited prediction CSV contains 11205 unique case-pattern-region records. Independent checks verified complete coverage, finite metrics and empty-mask rules; all 196 single-empty HD95 values match the corresponding evaluated-grid diagonals. Encoder-cache validation passed 30 real patch-pattern comparisons with bitwise-identical probabilities.
- Training-list SHA256: `1bf95a2c6c9983bac79e428df0c817a6c3cf8b19f6d13ca13d568ea4f6c2ee68`.
- Validation-list SHA256: `36572f11746985d79ef1c4bc8cade194b5b5b6b252a198ee6c7e399082ba8ac2`.
- Test-list SHA256: `e8f6a7d646ac5caa6a1160259774297a42fce6b010bff6afe4679d53474dc7b6`.
- Audited patient-metric CSV SHA256: `c89e06ab5f40d0047a71fd48d2b5f0fcc93f6704d719bcc30ae6fde97f78c270`.
- Current-partition evaluation-manifest SHA256: `a961ac442bfe0c228c724dd17a0b1b432c5a1fad27c0137dbab04ede3d7aa4bf`.
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
- Timing protocol: fixed input volume 240×240×155; 20 warmup / 50 timed passes; seed 20260727; FP32; TF32 off. This input belongs to efficiency profiling. Segmentation-record penalties use the per-case evaluated grids documented above. The timing settings are code-specified, not new measurements (`B:35–47,161–218`).
- Original reported 134.31 GFLOPs still requires the original command, source snapshot, input, enabled heads, precision/backend settings, and raw output.

</details>

## Source evidence

The identities and SHA256 hashes below identify architecture and transform source snapshots and the current-partition record-audit manifest. Those snapshots are not bundled here. Numeric source lines refer to the identified source snapshots; the current audit manifest is identified by its hash.

<details>
<summary>Source-file identities and SHA256 hashes</summary>

| Key | Source identity | SHA256 |
|---|---|---|
| AB | Archived `ablation.py` (not bundled) | `230fa07168244d0af5956f375de782a15e6170d78bb80632380bcc0db8845812` |
| B | Archived `benchmark_efficiency.py` (not bundled) | `b2d68ecfebc3084e49753558dffa33ee0aff9a916469ee69821ec85bd4d9d020` |
| D | Archived `datasets.py` (D) | `36f7ef36188cef3351a1bef789e6eaee51a678fb9a1c9270c487a9c3d52df176` |
| E | Archived `engine.py` (not bundled) | `dfe0d2566196a2661673babaea0e1fe276080510bc99359cc37e7302a0c182fa` |
| F | `evaluation_manifest.json` for the 249-case BraTS2023 record audit | `a961ac442bfe0c228c724dd17a0b1b432c5a1fad27c0137dbab04ede3d7aa4bf` |
| L | Archived `losses.py` (not bundled) | `489e7b0777b5354624a167736fd2a5a08069b1c9c63913e7497ddd94689d6173` |
| M | Archived `model.py` (not bundled) | `78280bae799bd8b113849dad1271568ecf78336b5790ff4ed44703ef07a80ebe` |
| N0 | Archived `transforms.py` (N0) | `706a695ae98b0f7b2de24b765ff6242b24718056d39750044ffeb66dbbf3cef7` |
| N1 | Archived `transforms.py` (N1) | `61746272820e188049280e835c06718a1b31ee309fc51a0fa9eeb6b77840118d` |
| P | Archived `presets.py` (not bundled) | `f8ddf66f507997c03a8da629966495d99870cee395464398d022254b7d1e5317` |
| Q | Archived `test.py` (Q) | `94fc5f34b6959c2e0fe74be3a2dde63864d07a23f1b90f232c76609edbf1eb9a` |
| R | Archived `服务器核查与启动命令_2026-10-26.md` (R) | `b70ba87eb61bda2b8b69a2c561cf357199c55f77701c16e986a249f6353063b9` |
| T | Archived `train_poly.py` (T) | `4da72cccba7b2a785b567da70d23dc8ad792e796beea687670389060975ced2c` |
| U | Archived `modules.py` (not bundled) | `d0198b99627470da4f709d555394ae1ac2dc4c7fdf9fa74f5e1f11cda3ac5759` |
| V | Archived `evaluate.py` (not bundled) | `41a70d3cfd1094e8f72883f23fff02c31e1fbca1d78540078e8fd6f33e187873` |

Evidence uses numeric source lines; the source identity and hash distinguish snapshots with the same filename.

</details>
