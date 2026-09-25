# Experimental Boundary Refinement

This optional inference-only candidate targets broad anomaly halos around thin
wafer2 scratches. It applies the grayscale guided filter from He et al., ECCV
2010, using the original image as guidance, and blends it with the existing map.
The default radius is 8 ORIGINAL-image pixels, epsilon 0.001 for [0,1] luminance,
and strength 0.5. These are starting parameters, not validated optimum values.

No retraining, GPU, masks, additional weights, or new dependencies are required.
NumPy, Pillow and SciPy are used. Only wafer2 maps change by default; all other
maps are copied byte-for-byte. CSV image scores are preserved verbatim. This
preserves Image F1 when downstream evaluation uses those scores, but not when
a submission system recomputes image scores from the refined maps.

```bash
python refine_omniad_maps.py --predictions ./predictions/hard3_routed --output_dir ./predictions/hard3_refined
python evaluate_omniad_export.py --predictions ./predictions/hard3_routed --output ./diagnostics/routed_reference.json
python evaluate_omniad_export.py --predictions ./predictions/hard3_refined --output ./diagnostics/refined_candidate.json
```

All commands default to data path `../dataset/download/Omni-AD-30-release`.
The destination must not already exist. Original inputs are never overwritten.
Keep baseline and candidate on the same image list and evaluation protocol.
`priority_objective = 25*pixel_f1 + 6*pixel_aupro + 18*image_f1` represents user
preferences, NOT official points. Previous `weighted_proxy_50` output is retired.
The old fusion config filename is retained for compatibility; it contains map
fusion coefficients, not metric priorities.

Guidance edges are not defect labels. Normal texture may be transferred to the
map, weak or invisible defects will not necessarily improve, and recall/AUPRO
can regress. There is no automatic replacement of baseline and no guaranteed
accuracy gain. Synthetic tests only verify mechanics. Compare all three metrics
on an authorized development set; do not tune on competition test labels.
If these images have already selected the routing policy, they are not an
independent estimate of generalization. Keep a separate holdout.

Reference: https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf
