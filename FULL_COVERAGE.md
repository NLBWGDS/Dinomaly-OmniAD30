# Legacy Crop Coverage Experiment

Current all30 development baseline: `diagnostics/all30_current/current_predictions`.
Measured means: Pixel F1 0.427451, Image F1 0.907122, AUPRO 0.711004.
Do not replace it before this candidate is evaluated.

The legacy model first resizes to a square `image_size`, then center-crops to
`crop_size`. Existing original-resolution export places the center crop back
into a zero canvas. Thus the model never sees the discarded border. This is
not merely a metric bug; full-image deployment genuinely lacks predictions
there. The share of real defects affected is unknown from summary metrics.

The new predictor loads geometry from the trusted original checkpoint and
refuses to run if it is not legacy or has no cropped border. It preserves the
same square resize scale, interpolation, normalization, encoder/decoder weights,
feature weights (0.5,0.5), Gaussian kernel 3/sigma 1, and crop input size used
by the freshly exported unified baseline. It adds shifted crop windows that
cover the resized canvas. For 448/392, center plus four corners yields five
views and covers the previously unseen 23.4375% of the resized image area.
This percentage is NOT the percentage of missed defects.

The original center prediction is retained inside its window; weighted shifted
predictions fill only outside. Bilinear resizing to original image coordinates
can change interpolated boundary pixels. Hard changes across the crop boundary
remain a possible artifact; this is a coverage ablation, not a guaranteed fix.
No old missing-border scores are synthesized by copying or extrapolating them.
The new scores come from actual model forwards on those regions.

First apply only to nameplate7, ceramic_wafer, spindle_top and chip1. The first
three have the lowest Pixel F1, and chip1 has the lowest AUPRO. All CSV image
scores are preserved. Existing successful hard3 maps are not modified. Only
the selected four categories are rerun; other maps are referenced unchanged.

```bash
python -u predict_omniad_fullcoverage.py --predictions ./diagnostics/all30_current/current_predictions --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth --output_dir ./predictions/all30_fullcoverage --batch_size 2
python evaluate_omniad_export.py --predictions ./predictions/all30_fullcoverage --output ./diagnostics/all30_fullcoverage.json
```

Both commands default to `../dataset/download/Omni-AD-30-release`. The export
requires the full30 source image set to match the dataset. Keep its source
directories because unchanged heatmaps are referenced, not copied. A new
output directory is required. The script never reads labels or modifies weights.
Evaluate only on the permitted development set, not private test labels.

Baseline category Pixel F1 / AUPRO:
- nameplate7: 0.037878 / 0.550119
- ceramic_wafer: 0.083761 / 0.331728
- spindle_top: 0.161643 / 0.771724
- chip1: 0.416957 / 0.227041

Image F1 should remain 0.907122 overall when using the copied CSV scores. Compare
all30 results against `diagnostics/all30_current/current_all30.json`, not old
training logs. No ground-truth crop restriction is applied to hide missing
regions. If the weak categories do not improve, border coverage is insufficient
to explain the failure; examine source images, masks and feature maps next.

Costs: normally five model forwards instead of one for these categories. The
manifest records actual windows, geometry, selected categories and branch time.
Reported time includes image decoding and prediction but excludes file export,
prior baseline generation and warmup. It is not official deployment latency.
Local tests cover geometry, center preservation, stitching and export integrity,
not real GPU accuracy. The full30 scheme remains a multi-checkpoint combination.
