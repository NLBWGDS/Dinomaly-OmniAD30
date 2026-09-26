# Normal Reference Experiment

Current best measured development candidate: `predictions/hard3_memory_bottle`.
Keep its parent `predictions/hard3_routed` for controlled branch comparisons.
The guided-filter experiment (`refine_omniad_maps.py`) regressed wafer2 pixel F1
from 0.410758 to 0.409475. Do not use `hard3_refined` as the new baseline.
Keep evaluation/diagnostic scripts. Keep the failed experiment and its tests for
reproducibility; it is opt-in and never loaded by the main inference entry point.

## Algorithm

`predict_omniad_memory.py` introduces a normal patch retrieval branch using the
existing Dinomaly encoder features. It builds a per-category bank from
`train/good` only, with seeded equal-per-image random sampling (8192 patches
maximum by default). Letterbox padding patches are excluded. Features are
normalized per group and weighted before concatenation. Exact cosine nearest
neighbor lookup is chunked to bound the similarity matrix allocation.

The concept is inspired by PatchCore (https://arxiv.org/abs/2106.08265), but this
is NOT a full reproduction: no greedy coreset, no separate CNN, no PatchCore
image-score reweighting. It adds a different source of anomaly evidence, not
another spatial smoothing filter. Unseen normal appearances can still trigger
false positives, and subtle defects may still be absent from encoder features.

To align the memory branch with reconstruction scores, a single scale is fit
from normal-image 95th percentiles. Every calibration query excludes ALL bank
patches from the same image. No test labels, test extrema, or ground-truth masks
are used to fit this scale. At least two normal images are required. The bank,
scale, seed and normal image list are saved in the output directory. The
checkpoint and seed must also be kept to reproduce this experiment.

Only bottle-bottom maps change in the first experiment. The fixed candidate is
75% current routed map + 25% scaled memory distance. This coefficient is an
experimental starting point, NOT a validated optimum. Other category maps and
ALL CSV image scores remain unchanged. Image F1 is therefore preserved only
when the evaluator consumes CSV scores independently of the pixel maps.

## Run

No gradient training or new packages are required. Use your existing GPU env:

```bash
python -u predict_omniad_memory.py --predictions ./predictions/hard3_routed --checkpoint ./saved_results/hard3_784/omniad_dinomaly_uni.pth --output_dir ./predictions/hard3_memory_bottle
python evaluate_omniad_export.py --predictions ./predictions/hard3_memory_bottle --output ./diagnostics/memory_bottle.json
```

Default data root is `../dataset/download/Omni-AD-30-release`. The output folder
must be new. One image is forwarded at a time. The first two passes build and
calibrate the normal bank; the final pass exports predictions. Use
`--query_chunk 64` to reduce similarity-matrix memory if necessary. This cannot
reduce encoder activation memory. `--device cpu` is available for debugging.

Compare with `diagnostics/routed_reference.json` using the same image list and
protocol. Preserve the three separate metrics; `priority_objective` with weights
25:6:18 is a preference objective, not official competition points. Evaluate on
the organizer-provided labeled development set only; final private test data
must not be used to select variants. This experiment makes no promised gain.

This prototype rebuilds the bank each run. It saves the artifact for auditing,
but does not yet provide a load-bank production mode. Timings in the manifest
are additional branch cost including IO, without warmup, excluding prior routed
inference. They are NOT total deployment latency. Nearest-neighbor retrieval and
per-category normal banks also require deployment-cost and organizer-rule review;
sharing encoder weights alone does not guarantee the single-model bonus.

## Measured Random-Bank Result and Coreset Candidate

On the supplied 43 bottle-bottom development images, the random-bank branch
raised pixel F1 from 0.335597 to 0.365916 and AUPRO from 0.911446 to 0.928418.
Image F1 remained 0.823529. Three-category mean pixel F1 rose from 0.413507 to
0.423613. This is a development-set result, not private-test or all-30-class
evidence. Preserve these predictions.

The next opt-in candidate uses a larger, bounded random normal-feature pool and
projected farthest-first coreset selection instead of direct random bank sampling.
The objective is to cover diverse normal appearances at approximately the same
retrieval bank size, reducing redundant normal features. It can also preserve
normal-data outliers, so improved detection is NOT guaranteed.

The first comparison retains the checkpoint, feature weights, 25% memory fusion
weight, normal-only calibration protocol and CSV image scores. Coreset selection
and its candidate pool change together. To isolate just selection in a later
ablation, both methods would need the same candidate pool. Defaults remain
`--sampling random` for backward compatibility.

```bash
python -u predict_omniad_memory.py --predictions ./predictions/hard3_routed --checkpoint ./saved_results/hard3_784/omniad_dinomaly_uni.pth --output_dir ./predictions/hard3_memory_coreset --sampling coreset
python evaluate_omniad_export.py --predictions ./predictions/hard3_memory_coreset --output ./diagnostics/memory_coreset.json
```

IMPORTANT: use `hard3_routed` as input again, NOT `hard3_memory_bottle`, which
already contains the random-bank contribution. Compare the new output against
`diagnostics/memory_bottle.json`. Do not stack two retrieval blends accidentally.

The default candidate pool is at most four times the previous bank budget;
Gaussian projection uses 64 dimensions only for selection. Retrieval retains
full-dimensional descriptors. No full candidate-by-candidate distance matrix
is built. Bank size remains 8183 for the supplied 49 normal images. Selection
adds bank-building time and memory; the manifest reports selection time. Exact
reproducibility across different GPU architectures is not guaranteed.

References inspected:
- User's index: https://github.com/M-3LAB/Awesome-Industrial-Anomaly-Detection
- PatchCore paper: https://arxiv.org/abs/2106.08265
- Official sampler: https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/sampler.py

Our Gaussian projection, single seeded start, bounded candidate pool and duplicate
suppression are implementation choices, not a claim to reproduce the official
sampler. No external industrial data or pretrained industrial weights are added.

## Detail Plus Neighborhood Candidate

Measured coreset bottle result: pixel F1 0.366931, AUPRO 0.921158, Image F1
0.823529. Compared to random selection, the small F1 gain did not offset AUPRO
loss under the requested 25:6:18 priorities. Keep `hard3_memory_bottle` as the
overall development baseline; keep coreset as a separate F1-leading candidate.

The next experiment retains random sampling, the same sampled patch positions,
and the 25% memory-map fusion. Only the retrieval descriptor changes. It
concatenates the original normalized descriptor with a separately normalized
3x3 neighborhood-average descriptor. Square-root weights assign 75% of their
dot product to center detail and 25% to context. This is feature-space context,
not another blur of the output anomaly map. It is inspired by local aggregation
in PatchCore but retains an explicit unsmoothed center branch, unlike a direct
replacement by pooled features. No claim of full reproduction is made.

Neighbor aggregation excludes letterbox padding via a geometry-only boolean
mask. The bank, queries and normal-only calibration all use the same new
descriptor. Calibration scale is necessarily refit; masks/labels are never used.
The original normal-image sampling and image-level scores remain unchanged.
Default `--context_weight 0` retains the previous behavior exactly. Bank feature
dimension doubles for nonzero context weight, increasing bank storage and
retrieval arithmetic. Similarity chunk size and encoder input size are unchanged.

```bash
python -u predict_omniad_memory.py --predictions ./predictions/hard3_routed --checkpoint ./saved_results/hard3_784/omniad_dinomaly_uni.pth --output_dir ./predictions/hard3_memory_context --sampling random --context_weight 0.25 --context_kernel 3
python evaluate_omniad_export.py --predictions ./predictions/hard3_memory_context --output ./diagnostics/memory_context.json
```

Again, do not feed an already memory-fused directory back into this experiment.
Compare with `memory_bottle.json` (F1 0.365916, AUPRO 0.928418). No retraining
is required. Context can also weaken subtle-defect discrimination; synthetic
tests validate implementation, not real accuracy. Retain the random baseline
unless authorized development evaluation supports replacement.
