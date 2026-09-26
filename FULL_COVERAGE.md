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

## Normal-Only Border Calibration Candidate

The supplied audit confirmed center score changes below 2.35e-7 and identical
center TP/FP/FN at the original threshold. Ceramic and chip normal border means
were approximately 2.55 times their center means. These are DEVELOPMENT
observations, not values used to fit the calibration below. Crop coverage misses
real defects, but raw border prediction also raises false positives substantially.

The opt-in `--normal_calibration` branch learns a per-category spatial normal
reference using ONLY `train/good`. Each normal image runs through the same
full-coverage predictor as inference. Area-averaged 32x32 grids bound storage;
at each grid cell, median and 95th percentile across normal images define local
offset and spread. The distribution of normal grid cells wholly inside the
original center defines a common reference median and spread. Quantile grids
are interpolated to the resized canvas before correcting border scores.

The affine gain is clipped to [0.25,4], corrected values are nonnegative, and
the first candidate uses 50% raw + 50% corrected border scores. Center scores
are copied back unchanged before original-resolution interpolation. No masks,
development thresholds, or per-test-image statistics are used in fitting.
At least five normal images and nondegenerate normal reference variation are
required. Test predictions never update the fit. These hyperparameters are
fixed experimental starting choices, not validated optimum values.

```bash
python -u predict_omniad_fullcoverage.py --predictions ./diagnostics/all30_current/current_predictions --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth --output_dir ./predictions/all30_calibrated_coverage --categories nameplate7,ceramic_wafer,spindle_top,chip1 --normal_calibration --calibration_strength 0.5 --batch_size 2
python evaluate_omniad_export.py --predictions ./predictions/all30_calibrated_coverage --output ./diagnostics/all30_calibrated_coverage.json
```

Use the accepted all30 baseline as input, NOT the rejected fullcoverage export.
Only four category maps are regenerated, but all30 are evaluated. Input maps
of the other26 categories and ALL CSV image scores remain unchanged. The
comparison baseline remains Pixel F1 0.427451, AUPRO 0.711004, Image F1 0.907122.
Also inspect each changed category; do not let mean AUPRO hide an F1 collapse.

Per-category `<category>_normal_calibration.npz` artifacts store normal quantile
grids, reference statistics, geometry and sample count. The coverage manifest
records normal image paths, arguments and fitting time. It never saves derived
development-label thresholds as deployment calibration. There is no additional
gradient training, but generating normal-image maps adds GPU runtime. Current
entry point refits calibration per run rather than providing an artifact-load
production mode. Preserve checkpoint, normal data, manifests and calibration
files to reproduce the experiment.

Position-dependent calibration assumes normal training images represent the
pose/edge distributions of deployment data. Pose changes or scarce samples can
make it harmful. It cannot recover defects absent from the features, and there
may still be score discontinuities at the old crop boundary. This is a tested
implementation of a hypothesis, NOT a measured improvement until all30
development results are available. Private test labels must not guide fitting.
