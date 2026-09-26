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

## Extend to Iron Lattice Without Reprocessing Bottle

Measured context-bottle result: pixel F1 0.370645, AUPRO 0.931648 and Image F1
0.823529. Keep `hard3_memory_context` as the current best measured development
baseline. Three-category mean pixel F1 is 0.425189, AUPRO 0.829434, Image F1
0.894045. These numbers are not evidence of private-test generalization.

The next controlled experiment applies the existing context retrieval branch to
iron_lattice only. Its baseline precision is 0.818907 but recall only 0.353845;
the objective is to recover missed defects without excessive false positives.
The same inference algorithm is reused; no new model or training loss is added.
Explicit feature weights 0.5,0.5 retain iron's original group weighting instead
of the script's bottle-oriented default 0.25,0.75.

```bash
python -u predict_omniad_memory.py --predictions ./predictions/hard3_memory_context --checkpoint ./saved_results/hard3_784/omniad_dinomaly_uni.pth --output_dir ./predictions/hard3_memory_context_iron --category iron_lattice --feature_weights 0.5,0.5 --sampling random --memory_weight 0.25 --context_weight 0.25 --context_kernel 3
python evaluate_omniad_export.py --predictions ./predictions/hard3_memory_context_iron --output ./diagnostics/memory_context_iron.json
```

Unlike earlier bottle variants, input now intentionally uses the successful
bottle export: the selected category is DIFFERENT. The exporter copies bottle
and wafer maps byte-for-byte, and all CSV image scores are unchanged. An
end-to-end synthetic regression covers this sequential workflow. Do not select
bottle again on this input, which would double-fuse that category.

Compare iron against pixel F1 0.494165, AUPRO 0.754967, Image F1 0.875. Compare
all three means against `memory_context.json`. A recall increase alone is not
enough; retain the previous baseline if F1 and the 25:6:18 objective regress.
This adds retrieval cost for iron, so record its manifest timing too. The new
output contains the new iron bank only; preserve the previous bottle output
and bank for reproducibility. No files need deletion.

## Scale-Matched Tiled Retrieval for Iron

The whole-image iron branch regressed: pixel F1 0.494165 -> 0.489626 and AUPRO
0.754967 -> 0.733501. Keep `hard3_memory_context`, NOT the `_iron` candidate.
Precision and recall both fell; the metrics alone do not establish the cause.

The next hypothesis is scale mismatch / insufficient local defect resolution.
The accepted iron reconstruction map uses tiled inference, but the attempted
memory branch used whole-image descriptors. New `--tile_grid 2` uses the SAME
2x2 overlapping crops for normal-bank fitting, normal calibration and inference.
Each crop is resized using checkpoint letterboxing. This increases effective
local sampling resolution, without introducing a new pretrained encoder or
using anomaly labels. It cannot recreate details absent from the source image.

The per-original-image bank quota remains unchanged. Calibration excludes all
bank patches from the same original image, not just the current crop. Crops
are restored and stitched with the existing positive Hann weights. Image scores
and unselected categories remain untouched. `--tile_grid 1` is still the default
and retains previous whole-image behavior. No global-view mixture is added to
the memory branch in this first scale experiment.

```bash
python -u predict_omniad_memory.py --predictions ./predictions/hard3_memory_context --checkpoint ./saved_results/hard3_784/omniad_dinomaly_uni.pth --output_dir ./predictions/hard3_memory_iron_tiles --category iron_lattice --feature_weights 0.5,0.5 --sampling random --memory_weight 0.25 --context_weight 0.25 --context_kernel 3 --tile_grid 2 --tile_overlap 0.25
python evaluate_omniad_export.py --predictions ./predictions/hard3_memory_iron_tiles --output ./diagnostics/memory_iron_tiles.json
```

Use the accepted `hard3_memory_context` as input, never the rejected iron output.
Compare iron against F1 0.494165, AUPRO 0.754967, Image F1 0.875 and recall
0.353845. Preserve bottle F1 0.370645 and wafer F1 0.410758. Keep the prior
baseline unless development results justify replacement.

There are normally four model forwards per image per pass instead of one.
Views are processed sequentially, but CPU descriptor storage during fitting
and total runtime increase. This experimental exporter also decodes each crop
from the source separately; its timings include that overhead. No deployment
latency claim is made. Bank artifacts record crop configuration. Actual GPU
accuracy/latency must be measured on the server; local tests use synthetic data.
# Legacy full-frame retrieval experiment

The latest accepted development candidate is `predictions/all30_detail_candidate`,
not `all30_detail_gated`. Gating reduced mean Pixel F1 from 0.4337489939 to
0.4328038708 and is not adopted. Mean AUPRO for the accepted candidate is
0.7472751807 and Image F1 is 0.9071216954.

The existing normal-reference branch now supports an explicit
`--legacy_full_frame` option for the original unified checkpoint. Each view is
resized to the checkpoint crop size without center cropping, so its borders
remain visible. All feature patches are valid; unlike letterbox mode there is
no padding. Original checkpoint weights and metadata on disk are unchanged.
The manifest and bank artifact record original and inference geometries.
This is an inference geometry change, not evidence of a trained-model gain.

This experiment adds normal feature retrieval to ceramic_wafer, rather than
reweighting the same pair of reconstruction maps again. Normal bank fitting,
leave-one-original-image-out scale calibration and test inference use identical
view geometry. All crops belonging to the calibration query image are excluded.
No development masks enter bank fitting or inference. The coreset implementation
and calibration are the existing repository algorithms, not a new PatchCore model.

Run from the server project directory:

```bash
python -u predict_omniad_memory.py \
  --predictions ./predictions/all30_detail_candidate \
  --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth \
  --data_path ../dataset/download/Omni-AD-30-release \
  --output_dir ./predictions/all30_ceramic_memory \
  --category ceramic_wafer \
  --legacy_full_frame --require_all30 --reference_unselected \
  --tile_grid 2 --tile_overlap 0.25 \
  --feature_weights 0.5,0.5 --context_weight 0 \
  --sampling coreset --bank_size 8192 --memory_weight 0.25

python evaluate_omniad_export.py \
  --predictions ./predictions/all30_ceramic_memory \
  --output ./diagnostics/all30_ceramic_memory.json
```

This requires GPU inference and offline bank fitting, but no gradient training
or additional dependencies. Keep all source prediction directories. The other
29 categories and every image score are unchanged. `--require_all30` checks the
complete dataset image inventory before fitting. Existing output directories
are rejected. Default legacy behavior still rejects unapproved geometry changes;
existing letterbox memory commands do not change.

Compare full30 metrics with the accepted candidate, and ceramic Pixel F1 with
0.0898267656 / AUPRO 0.6400206130. Keep this as a separate experiment until an
actual improvement is measured. Normal retrieval can miss anomalous patches
similar to normal ones; square resizing can distort aspect ratio; spatially
unrestricted retrieval may confuse normal textures. There is no guarantee of
better Pixel F1. It adds four views and patch retrieval per selected test image;
latency must be measured before submission. Retain an independent holdout rather
than repeatedly tuning parameters to the same development labels.
