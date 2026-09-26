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

## Cached reliability-gated candidate

The measured fixed-detail experiment raised mean Pixel F1 to 0.4337489939 and
AUPRO to 0.7472751807, with Image F1 unchanged at 0.9071216954. Ceramic Pixel F1
was 0.0898267656, spindle Pixel F1 was 0.1717593788. Spindle precision rose but
recall at its best-F1 operating point fell. These observations do not prove which
individual pixels caused the change; the next experiment tests a fusion hypothesis.

`reblend_omniad_detail.py` reuses the fixed-detail export without model inference.
It computes a grid from normal training statistics:
`reliability = clip(reference_span / max(local_span, reference_span), 0.25, 1)`.
Here span is q95 minus median. More variable normal locations receive less weight.
The fixed-detail correction is `detail_export - original_baseline`; positive
corrections are multiplied by reliability, negative ones by reliability * 0.5.
Thus each new pixel lies between its original-baseline and fixed-detail scores.
No branch reconstruction, binary masks, new test-fitted statistics, or labels
are needed. This proxy is NOT a calibrated confidence probability.

This may reduce normal-texture boosts and retain coarse evidence lost during
detail blending. It can also restore baseline false positives or weaken real
defects in variable normal regions; a gain is not guaranteed. Image F1 remains
unchanged intentionally. This ablation does not improve the underlying model.

```bash
python reblend_omniad_detail.py \
  --baseline ./predictions/all30_calibrated_coverage \
  --detail_predictions ./predictions/all30_detail_candidate \
  --output_dir ./predictions/all30_detail_gated

python evaluate_omniad_export.py \
  --predictions ./predictions/all30_detail_gated \
  --output ./diagnostics/all30_detail_gated.json
```

Run from the same project directory as the original export: older manifests
record relative source paths. `--baseline` MUST be the original source recorded
in the fixed-detail manifest, not the already blended candidate. Keep both source
directories intact. The script validates source provenance, all30 image inventory,
normal calibration statistics and image-score agreement. It does not hash cached
maps, so do not modify the source exports in place. It rejects existing output
directories. The unchanged 28 categories reference the fixed-detail candidate.
Compare against the fixed-detail report above, not only the older coarse result.
Do not sweep these parameters repeatedly against the same labeled development set.
