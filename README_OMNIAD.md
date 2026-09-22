# Dinomaly Omni-AD Reproduction Notes

This repository originally targets public IAD benchmarks such as MVTec AD, VisA, Real-IAD, and MPDD. For JBGS-2026-08, use the Omni-AD entry point added in this fork:

```bash
python dinomaly_omniad_uni.py --mode train --data_path ../dataset/Omni-AD-30-release
```

## Compliance Boundary

- Training data: only reads `Omni-AD-30-release/<category>/train/good`.
- Public backbone: default is official DINOv2 register ViT-B/14 from Meta public files. Do not replace it with community weights that were self-supervised or fine-tuned on industrial anomaly data.
- Development test data: can be used for model selection, threshold/post-processing choices, and reporting. It is not used in the optimization loop.
- Final test/prediction: `predict` mode runs pure forward inference under `torch.no_grad()` and `model.eval()`. It does not update parameters, BatchNorm statistics, memory banks, or other model state.
- Do not run the original MVTec/VisA/Real-IAD/MPDD scripts for competition training; those datasets are outside the Omni-AD whitelist.

## Commands

Check that the local dataset and imports are ready without loading the backbone:

```bash
python dinomaly_omniad_uni.py --mode check
```

Train a single multi-category model:

```bash
python dinomaly_omniad_uni.py ^
  --mode train ^
  --data_path ../dataset/Omni-AD-30-release ^
  --output_dir ./saved_results/omniad_dinomaly_uni
```

Disable development-set evaluation during training:

```bash
python dinomaly_omniad_uni.py ^
  --mode train ^
  --data_path ../dataset/Omni-AD-30-release ^
  --eval_every 0
```

Evaluate a saved checkpoint on the labeled development test split:

```bash
python dinomaly_omniad_uni.py ^
  --mode eval ^
  --data_path ../dataset/Omni-AD-30-release ^
  --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth
```

Run pure-forward prediction and export anomaly scores plus `.npy` heatmaps:

```bash
python dinomaly_omniad_uni.py ^
  --mode predict ^
  --data_path ../dataset/Omni-AD-30-release ^
  --checkpoint ./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth ^
  --output_dir ./predictions/omniad_dinomaly_uni
```

Limit to a subset of categories while debugging:

```bash
python dinomaly_omniad_uni.py ^
  --mode train ^
  --data_path ../dataset/Omni-AD-30-release ^
  --categories air_conditioner_filter,battery_piece ^
  --total_iters 100
```

## Outputs

- Training checkpoint: `saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth`
- Training log: `saved_results/omniad_dinomaly_uni/log.txt`
- Prediction score file: `predictions/omniad_dinomaly_uni/scores.csv`
- Prediction heatmaps: one `.npy` file per test image, resized back to the input image size.

## Next Packaging Steps

1. Put allowed public backbone weights in `backbones/weights/` or allow the Docker build/runtime to download from the official source permitted by the platform.
2. Add the platform-required train and inference shell wrappers once the organizer publishes the exact command contract.
3. In the technical report, disclose the default backbone and state that no external industrial anomaly detection data is used.
