# Controlled map comparison

Run on an authorized labeled development set. This uses existing exports and
does not load a model or train. All three variants use identical original-size
masks. Image scores are recomputed from the top 1% of original-size map pixels
for all variants, rather than mixing potentially different CSV score protocols.

```bash
python compare_omniad_maps.py \
  --baseline ./predictions/hard3_784 \
  --candidate ./predictions/hard3_tiled \
  --output ./diagnostics/hard3_comparison.json
```

The output contains baseline, tiled candidate, and a fixed 50/50 blend. Pixel
F1 takes priority. Image F1 and AUPRO reveal tradeoffs. No best variant is
automatically selected from evaluation labels. Scores are not normalized per
image: global/local scale mismatch remains a potential limitation of fusion.

Pixel F1 uses 4096 common score bins per category; image F1 searches exact score
thresholds. AUPRO uses 8-connected components with equal per-region weighting,
and integrates the histogram PRO curve to FPR=0.3 with endpoint interpolation.
This is a documented development metric, not a claim of official evaluator
equivalence. Do not compare its AUPRO directly with the older resized evaluator.
Memory use scales with one image plus histograms, not the full image collection.

Optional `--export_blend_dir ./predictions/hard3_blend` exports fixed-weight maps
and scores. Export does not apply a threshold selected from labels. It uses a new
directory and does not overwrite existing prediction files.

Tests: `python -m unittest test_compare_omniad_maps -v`.
