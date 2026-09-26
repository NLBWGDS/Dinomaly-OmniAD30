# Full 30-Category Development Audit

Historical console metrics are NOT the latest combined-system evaluation.
This workflow measures both the unified checkpoint and the current fixed
category-routed candidate on exactly the same complete evaluation image set.
It does not train models or automatically search category policies.

## Fixed Policy

| Categories | Unified reference | Current candidate |
| --- | --- | --- |
| 27 categories outside hard3 | Original all30 checkpoint, fresh full-image export | Same exported maps and image scores |
| infusion_bottle_bottom5 | Same all30 checkpoint | Accepted `hard3_memory_context` export |
| iron_lattice | Same all30 checkpoint | Accepted `hard3_memory_context` export (original tiled routing, no added memory branch) |
| wafer2 | Same all30 checkpoint | Accepted `hard3_memory_context` export (original fixed map blend) |

The combined candidate uses multiple checkpoints and category routing. It must
NOT be reported as a single unified model. Unverified hard5/hard3 variants and
rejected iron experiments are not applied to the remaining categories. Changing
code did not retroactively change any checkpoint or previously exported map.

## One Command on the GPU Server

Run from the repository root after syncing code. Keep the accepted three-category
predictions and the original all30 checkpoint available at their existing paths:

```bash
python -u evaluate_omniad_all30.py --data_path ../dataset/download/Omni-AD-30-release --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth --overrides ./predictions/hard3_memory_context --output_dir ./diagnostics/all30_current --batch_size 2 --num_workers 4
```

The script checks for exactly 30 dataset categories and all evaluation images.
It checks checkpoint category metadata before launching fresh all30 inference;
a hard3-only checkpoint is rejected. Unified inference uses checkpoint geometry,
feature weights 0.5,0.5, Gaussian kernel 3/sigma 1, and image top ratio 0.01.
This is a freshly measured reference, not a claim to reproduce every historical
training-log option. All arguments and the executed command are saved.

The accepted hard3 maps and CSV scores are reused, not recomputed. They already
include the selected tiling, fusion and bottle normal-memory changes. Thus no
GPU retraining or repeated hard3 branch fitting is necessary. To reconstruct
those exports from scratch, retain the earlier branch scripts and manifests.

Then both complete systems are measured with `evaluate_omniad_export.py` logic:
original image coordinates, 4096-bin maximum pixel F1, AUPRO@FPR0.3 with
8-connected regions, and maximum image F1 from the actual CSV scores. The
priority objective 25:6:18 is not official competition points. Different image
score generation between algorithms is preserved as part of their definition;
the evaluator and image lists are identical.

## Outputs

- `unified_all30.json`: newly measured all30 unified-model reference.
- `current_all30.json`: all30 fixed combined-system results, including source per category.
- `comparison.csv`: all categories sorted by CURRENT Pixel F1 ascending, baseline/current/delta for each metric, and macro means.
- `current_predictions/sources.json`: fixed category-source policy and image count.
- `run.json`: arguments, executed inference command, category list and image count.
- `unified_predictions/`: fresh all30 heatmaps; these can be large.

Send `current_all30.json` and `comparison.csv` for the next analysis. Do not use
the old 30-category table or a hard3-only mean as substitutes.

The combined prediction CSV references existing heatmaps by absolute path to
avoid duplicating large files. Keep both source prediction directories. Reports
are self-contained, but the CSV alone is not a portable prediction package.
Missing/extra/duplicate image entries, shared map paths for different images,
nonfinite scores, invalid map shapes or missing masks cause errors rather than
silently reducing the evaluated set. No output directory is overwritten.

If inference completed but evaluation was interrupted, use a NEW report folder
and reuse the completed export with `--baseline_predictions`:

```bash
python -u evaluate_omniad_all30.py --baseline_predictions ./diagnostics/all30_current/unified_predictions --overrides ./predictions/hard3_memory_context --output_dir ./diagnostics/all30_retry
```

Reuse verifies image coverage, not the training history of an external export;
the caller must supply the intended unified baseline. Re-evaluation reads large
pixel maps twice per system, so it can take time on CPU. GPU latency, memory
costs and single-model eligibility are not measured by this audit. This is
authorized labeled-development evaluation, not private-test tuning.
