# Local-resolution experiment

Use the original hard3 checkpoint. No retraining or ground-truth access is
performed by the inference script. Default geometry is a 2x2 grid with 25%
overlap. Each crop uses checkpoint letterbox preprocessing. Hann blending
reduces tile seams; the final map contains 75% local and 25% full-image scores.
Scores are not independently normalized per tile.

```bash
python predict_omniad_tiled.py --mode predict \
  --checkpoint ./saved_results/hard3_784/omniad_dinomaly_uni.pth \
  --categories wafer2,infusion_bottle_bottom5,iron_lattice \
  --output_dir ./predictions/hard3_tiled --batch_size 2 \
  --category_feature_weights 'wafer2=0.25,0.75;infusion_bottle_bottom5=0.25,0.75'
python diagnose_omniad.py --predictions ./predictions/hard3_tiled \
  --output_dir ./diagnostics/hard3_tiled --top_k 5
```

Compare the pixel precision/recall/F1 in the original and tiled diagnostic
reports. Both must cover exactly the same images and use the same bins and
original-resolution masks. Diagnostic threshold searches use labels and are
only for an authorized development split, not final-test threshold selection.
The diagnostic report does not compute image F1 or AUPRO; pixel improvements
alone do not establish improvement of those metrics or the official score.

`--tile_grid 1 --global_weight 0` provides a full-image control using the same
export and image-score calculation. Use a separate output directory for it.

Potential costs: five views instead of one, increased inference time, changed
crop context, and incompatible score scales between global/local views. A
gain is not guaranteed. Keep existing checkpoints and predictions. The timing
JSON is diagnostic (no warmup, excludes file export), not official latency.

Tests: `python -m unittest test_omniad_tiling -v`.
