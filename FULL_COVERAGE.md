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

## Measured Result and Regression Audit

Full replacement is NOT accepted: all30 Pixel F1 fell from 0.427451 to 0.423512,
while AUPRO rose from 0.711004 to 0.741076. Image F1 remained 0.907122. The
other 26 categories were unchanged. Nameplate F1 improved to 0.130264 and
spindle F1 to 0.165306, but ceramic F1 fell to 0.004889 and chip F1 to 0.281608.
Keep both exports; do not promote this whole candidate or cherry-pick results
without another complete30 evaluation and clear development-selection disclosure.

The next step is diagnosis, not another algorithm change:

```bash
python diagnose_omniad_coverage.py --baseline ./diagnostics/all30_current/current_predictions --candidate ./predictions/all30_fullcoverage --output_dir ./diagnostics/coverage_audit --top_k 3
```

This requires only existing predictions and labeled development data, no GPU
inference. It validates the complete 30-category image inventory, then analyzes
the four changed categories. It does NOT recalculate unchanged categories or
claim a new full30 score. Actual crop geometry comes from coverage_manifest.json.

Three passes compute score ranges, histogram maximum-F1 thresholds and regional
counts. Regions are original-image pixels whose bilinear interpolation support
lies wholly within the old center, wholly outside it, or straddles its boundary.
For each source, counts are reported at its own category threshold. Candidate
counts are ALSO evaluated at the baseline threshold to separate score changes
from threshold movement. Region counts use these global thresholds, not
separately optimized regional thresholds. Center score mean/max absolute change
and number of changes exceeding 1e-5 test the center-preservation assumption.
Small floating-point differences may come from GPU batch arithmetic; large
changes require checking checkpoint, preprocessing, weights and map provenance.

Outputs:
- `coverage_audit/summary.json`: all four diagnostic reports, the file to send back.
- `<category>/per_image.csv`: per-image regional FP, ground-truth coverage and score differences.
- `<category>/01_baseline.png`, `01_candidate.png` (up to rank03): paired panels for images with the largest FP increases at each source's own threshold.
- `<category>/panels.json`: image identities corresponding to panel ranks.

Panels share one heatmap color range across both sources within each category;
error overlays use their respective diagnostic thresholds. These selected images
are not a representative sample of model performance. Thresholds depend on
development labels and must not be treated as deployable calibration. To check
histogram resolution sensitivity, a later run may use a new output directory
and `--bins 65536`; the default4096 matches the current evaluator.
