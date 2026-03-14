# HRSC2016-MS OBB Training Summary

## Overview

This repository trains YOLO OBB models on the processed HRSC2016-MS dataset and tracks experiment outputs under `runs/obb/`.

- Dataset config: `processed_obb/dataset.yaml`
- Train outputs: `runs/obb/runs_yolo_obb/expXX/`
- Test outputs: `runs/obb/runs_eval_obb/test_expXX/`
- Versioned lightweight artifacts: `artifacts/expXX/`

## Experiment Summary

The table below summarizes the five training runs completed so far.

| Exp | Model source | Pretrained | Main changes from previous run | Epochs recorded | Best train mAP50 | Best train mAP50-95 | Test mAP50 | Test mAP50-95 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| exp01 | `yolo11n-obb.yaml` | No | Baseline nano architecture, default augmentation bias | 150 | 0.84586 | 0.63890 | 0.72745 | 0.53514 |
| exp02 | `yolo11n-obb.yaml` | No | Longer training, lower LR, cosine LR, lighter geometry, mixup added | 350 | 0.85019 | 0.65126 | 0.76033 | 0.57438 |
| exp03 | `yolo11s-obb.yaml` | No | Model scaled from nano to small, lower LR, reduced augmentation strength | 292 | 0.84741 | 0.64421 | 0.68383 | 0.49955 |
| exp04 | `yolo11s-obb.pt` | Yes | Small model with pretrained weights, conservative LR, same light augmentation | 250 | 0.91342 | 0.73890 | 0.76788 | 0.60135 |
| exp05 | `yolo11m-obb.pt` | Yes | Larger pretrained model, lower LR, smaller batch, stronger scale/translate, lighter mosaic | 260 | 0.91642 | 0.73374 | 0.76781 | 0.60690 |

Notes:

- Current `exp02` directory contains the run originally created as `exp022` and later renamed.
- `Epochs recorded` reflects the actual rows saved in `results.csv`, not only the configured maximum epoch count.
- Test metrics come from `runs/obb/runs_eval_obb/test_expXX/test_metrics_summary.json`.
- The repository tracks compact experiment artifacts under `artifacts/expXX/` instead of committing the full `runs/` tree.

## Detailed Run Configuration

### exp01

- Train dir: `runs/obb/runs_yolo_obb/exp01`
- Test dir: `runs/obb/runs_eval_obb/test_exp01`
- Model: `yolo11n-obb.yaml`
- Pretrained weights: `False`
- Key hyperparameters:
  - `epochs=150`
  - `batch=8`
  - `imgsz=1024`
  - `lr0=0.01`
  - `cos_lr=False`
  - `weight_decay=0.0005`
  - `patience=30`
  - `degrees=0.0`
  - `translate=0.1`
  - `scale=0.5`
  - `mosaic=1.0`
  - `mixup=0.0`

Training metrics:

- Final: `precision=0.87319`, `recall=0.72248`, `mAP50=0.84542`, `mAP50-95=0.63890`
- Best train: `mAP50=0.84586` at epoch `149`, `mAP50-95=0.63890` at epoch `151`
- Final losses: `train_loss_sum=2.60705`, `val_loss_sum=3.02063`

Test metrics:

- `precision=0.80825`
- `recall=0.59698`
- `mAP50=0.72745`
- `mAP50-95=0.53514`

### exp02

- Train dir: `runs/obb/runs_yolo_obb/exp02`
- Test dir: `runs/obb/runs_eval_obb/test_exp02`
- Original run name in `args.yaml`: `exp022`
- Model: `yolo11n-obb.yaml`
- Pretrained weights: `False`
- Main changes vs exp01:
  - `epochs: 150 -> 350`
  - `batch: 8 -> 6`
  - `lr0: 0.01 -> 0.003`
  - `cos_lr: False -> True`
  - `patience: 30 -> 80`
  - `degrees: 0.0 -> 5`
  - `translate: 0.1 -> 0.05`
  - `scale: 0.5 -> 0.3`
  - `mosaic: 1.0 -> 0.7`
  - `mixup: 0.0 -> 0.03`
  - `close_mosaic: 10 -> 20`

Training metrics:

- Final: `precision=0.88961`, `recall=0.73425`, `mAP50=0.84008`, `mAP50-95=0.64607`
- Best train: `mAP50=0.85019` at epoch `234`, `mAP50-95=0.65126` at epoch `291`
- Final losses: `train_loss_sum=2.01594`, `val_loss_sum=3.01612`

Test metrics:

- `precision=0.81411`
- `recall=0.64317`
- `mAP50=0.76033`
- `mAP50-95=0.57438`

Interpretation:

- Best generalization among the first three runs.
- Longer training and lighter augmentation improved test performance over the baseline.

### exp03

- Train dir: `runs/obb/runs_yolo_obb/exp03`
- Test dir: `runs/obb/runs_eval_obb/test_exp03`
- Model: `yolo11s-obb.yaml`
- Pretrained weights: `False`
- Main changes vs exp02:
  - Model scaled from `yolo11n-obb.yaml` to `yolo11s-obb.yaml`
  - `epochs: 350 -> 300`
  - `batch: 6 -> 4`
  - `lr0: 0.003 -> 0.002`
  - `weight_decay: 0.0005 -> 0.0007`
  - `patience: 80 -> 50`
  - `degrees: 5 -> 3`
  - `translate: 0.05 -> 0.04`
  - `scale: 0.3 -> 0.25`
  - `mosaic: 0.7 -> 0.5`
  - `mixup: 0.03 -> 0.0`
  - `close_mosaic: 20 -> 25`

Training metrics:

- Final: `precision=0.86582`, `recall=0.74296`, `mAP50=0.84271`, `mAP50-95=0.63901`
- Best train: `mAP50=0.84741` at epoch `224`, `mAP50-95=0.64421` at epoch `243`
- Final losses: `train_loss_sum=1.87925`, `val_loss_sum=3.05798`

Test metrics:

- `precision=0.80755`
- `recall=0.52063`
- `mAP50=0.68383`
- `mAP50-95=0.49955`

Interpretation:

- Increasing model size without pretrained weights did not improve generalization.
- This run shows the clearest gap between train metrics and test metrics.

### exp04

- Train dir: `runs/obb/runs_yolo_obb/exp04`
- Test dir: `runs/obb/runs_eval_obb/test_exp04`
- Model: `yolo11s-obb.pt`
- Pretrained weights: `True`
- Main changes vs exp03:
  - Model source switched from architecture-only `yolo11s-obb.yaml` to pretrained `yolo11s-obb.pt`
  - `epochs: 300 -> 250`
  - `lr0: 0.002 -> 0.0015`
  - `weight_decay: 0.0007 -> 0.0005`
  - `close_mosaic: 25 -> 20`
  - Other core augmentation settings kept similar

Training metrics:

- Final: `precision=0.90873`, `recall=0.84383`, `mAP50=0.90824`, `mAP50-95=0.73710`
- Best train: `mAP50=0.91342` at epoch `216`, `mAP50-95=0.73890` at epoch `217`
- Final losses: `train_loss_sum=1.34591`, `val_loss_sum=2.68375`

Test metrics:

- `precision=0.92095`
- `recall=0.59544`
- `mAP50=0.76788`
- `mAP50-95=0.60135`

Interpretation:

- Best overall run so far.
- Pretrained small model delivered the highest test `mAP50` and test `mAP50-95`.
- Test recall is still lower than the train recall trend, so recall-focused tuning remains possible.

### exp05

- Train dir: `runs/obb/runs_yolo_obb/exp05`
- Test dir: `runs/obb/runs_eval_obb/test_exp05`
- Model: `yolo11m-obb.pt`
- Pretrained weights: `True`
- Main changes vs exp04:
  - Model scaled from `yolo11s-obb.pt` to `yolo11m-obb.pt`
  - `epochs: 250 -> 260`
  - `batch: 4 -> 2`
  - `workers: 4 -> 2`
  - `lr0: 0.0015 -> 0.0010`
  - `weight_decay: 0.0005 -> 0.00035`
  - `patience: 50 -> 40`
  - `translate: 0.04 -> 0.08`
  - `scale: 0.25 -> 0.35`
  - `mosaic: 0.5 -> 0.35`
  - `close_mosaic: 20 -> 15`

Training metrics:

- Final: `precision=0.91976`, `recall=0.83308`, `mAP50=0.91188`, `mAP50-95=0.73274`
- Best train: `mAP50=0.91642` at epoch `118`, `mAP50-95=0.73374` at epoch `254`
- Final losses: `train_loss_sum=1.45181`, `val_loss_sum=2.69084`

Test metrics:

- `precision=0.92493`
- `recall=0.59175`
- `mAP50=0.76781`
- `mAP50-95=0.60690`

Interpretation:

- `mAP50` is effectively tied with `exp04`, but `mAP50-95` improved slightly.
- Precision is the highest among all runs, while recall remains moderate.
- Moving from `s` to `m` did not create a meaningful gain in `mAP50`, so the larger model mostly improved stricter IoU performance rather than broad recall.

## Conclusions

1. `exp01` established a stable baseline with nano architecture and no pretrained weights.
2. `exp02` improved generalization by tuning schedule and augmentation while staying on the nano architecture.
3. `exp03` showed that a larger architecture alone is not enough on this dataset when trained from scratch.
4. `exp04` confirmed that pretrained weights matter more than architecture size alone for this project.
5. `exp05` showed that scaling from pretrained `s` to pretrained `m` gives only marginal gains on this dataset, mainly in `mAP50-95`, not in headline `mAP50`.

## Recommended Current Model

Use the checkpoint based on the metric you care about most:

- Best test `mAP50`: `exp04` with `0.76788`
- Best test `mAP50-95`: `exp05` with `0.60690`
- Best test precision: `exp05` with `0.92493`

Practical recommendation:

- Use `runs/obb/runs_yolo_obb/exp04/weights/best.pt` if you prioritize overall balance and slightly better recall.
- Use `runs/obb/runs_yolo_obb/exp05/weights/best.pt` if you prioritize stricter localization quality and precision.

## Versioned Artifacts

Each `artifacts/expXX/` folder contains the compact files kept in git for sharing and review:

- `args.yaml`
- `train_config.json` when available
- `results.csv`
- `results.png`
- `loss_curve.png`
- `accuracy_curve.png`
- `BoxPR_curve.png`
- `BoxF1_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `epoch_metrics.json`
- `test_metrics_summary.json`

## Reference Files

- `artifacts/exp01/args.yaml`
- `artifacts/exp02/args.yaml`
- `artifacts/exp03/args.yaml`
- `artifacts/exp04/args.yaml`
- `artifacts/exp05/args.yaml`
- `artifacts/exp01/results.csv`
- `artifacts/exp02/results.csv`
- `artifacts/exp03/results.csv`
- `artifacts/exp04/results.csv`
- `artifacts/exp05/results.csv`
- `artifacts/exp01/test_metrics_summary.json`
- `artifacts/exp02/test_metrics_summary.json`
- `artifacts/exp03/test_metrics_summary.json`
- `artifacts/exp04/test_metrics_summary.json`
- `artifacts/exp05/test_metrics_summary.json`
