# Normal-calibrated detail candidate

This is an optional experiment, not a replacement for accepted predictions.
It uses the same unified checkpoint. No retraining or new dependencies are needed.

The legacy coverage branch fills borders but keeps the original resize scale.
`--detail_scale 2` instead doubles the square canvas while keeping each model
input crop unchanged (typically 896 canvas / 392 crop). Nine overlapping windows
are stitched with positive Hann weights, including the center window. This gives
more input pixels to small defects without increasing per-view token count.
It does not restore details absent from the original image.

For each selected category, train/good images alone fit spatial median and q95
grids at both scales. Detail scores are calibrated across the entire canvas to
the coarse normal reference distribution, with bounded gains, then resized
directly to original image dimensions and mixed with the source prediction.
No labels, test-image quantiles, or binary morphology are used in this operation.
The original-resolution source map contributes 75% at detail_weight=0.25.
`calibration_strength` affects only the legacy border mode, not detail mode.

All image scores and unselected maps are preserved. Source prediction directories
must remain available because exports reference their unchanged maps by path.
The supplied predictions must cover the full 30-category dataset.

## Server experiment

Use the recently evaluated calibrated all30 export as the source, not hard3.
The checkpoint below must be the original all30 legacy checkpoint.

```bash
python -u predict_omniad_fullcoverage.py \
  --predictions ./predictions/all30_calibrated_coverage \
  --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth \
  --data_path ../dataset/download/Omni-AD-30-release \
  --output_dir ./predictions/all30_detail_candidate \
  --categories ceramic_wafer,spindle_top \
  --normal_calibration --detail_scale 2 --detail_weight 0.25 --batch_size 2

python evaluate_omniad_export.py \
  --predictions ./predictions/all30_detail_candidate \
  --output ./diagnostics/all30_detail_candidate.json
```

Choose a fresh output directory on repeat runs. This exports all30 while only
recomputing the two selected categories. Normal fitting is performed before test
inference and saved as per-category calibration NPZ files and manifest paths.

## Acceptance

Compare the entire report against both the accepted old all30 baseline and the
latest calibrated export. The latter mean is Pixel F1=0.4324081995,
AUPRO=0.7462765187, Image F1=0.9071216954. Its ceramic Pixel F1=0.0577752507
is still below the older baseline 0.0837612485; recovery relative to the latest
candidate alone is insufficient evidence of fixing that regression.

Prioritize Pixel F1; report AUPRO and Image F1 alongside the 25:6:18 preference
objective (not official points). The other 28 categories and every Image F1
should remain identical. Do not automatically accept or overwrite either source.
Use an independent holdout to check repeated development-set selection.

Scale changes can also amplify normal texture or lose global context. No measured
gain is claimed until server evaluation. This costs 9 views per selected test
image plus offline normal fitting (typically 14 views per normal image). The
manifest timing excludes source inference, export and fitting; it is not proof
of meeting the competition latency limit. The complete routed pipeline still
uses multiple checkpoints, even though this branch reuses the unified model.
