# Normal Reference Experiment

Current accepted baseline: `predictions/hard3_routed`.
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
