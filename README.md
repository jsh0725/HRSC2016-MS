# HRSC2016-MS OBB 선박 탐지 프로젝트 정리

## 1. 프로젝트 목적
- HRSC2016-MS 데이터셋을 OBB(회전 바운딩 박스) 형식으로 전처리
- YOLO OBB 모델(`yolo11n-obb.yaml`) 학습
- 학습 결과(지표 + 시각화 이미지)를 정리하여 비교/분석 가능 상태로 구성

## 2. 현재 폴더 구조
| 경로 | 설명 |
|---|---|
| `archive/` | 원본 데이터(BMP, XML, split txt) | -> git에는 미업로드
| `processed_obb/` | OBB 전처리 결과(`images/`, `labels/`, `dataset.yaml`) | -> git에는 미업로드
| `runs/obb/runs_yolo_obb/exp01/` | 학습(Train)산출물 폴더, 에폭별 학습로그 + 학습 검증 시각화 + PR/F1 곡선 + confusion matrix + 샘플 배치 이미지 + weight 포함 |
| `runs/obb/runs_eval_obb/test_exp01/` | 학습 완료 모델의 평가/추론 산출물 폴더, 테스트셋 정량 지표 + 테스트 이미지 예측 시각화(예측 박스 오버레이) 포함 |
| `results/exp01/` | 중요한 PNG 시각화 파일만 재정리한 폴더 |
| `processing_obb.py` | OBB 전처리 스크립트 |
| `train_yolo_obb.py` | YOLO OBB 학습 + epoch 메트릭/곡선 저장 |
| `eval_test_yolo_obb.py` | 테스트셋 평가/시각화 스크립트 |

## 3. 실행 순서
```bash
python processing_obb.py
python train_yolo_obb.py
python eval_test_yolo_obb.py
```

## 4. 전처리 결과 요약
기준 파일: `processed_obb/summary.json`

| Split | 입력 ID | 출력 이미지 | 출력 라벨 | 객체 수 | 누락/오류 |
|---|---:|---:|---:|---:|---:|
| train | 610 | 610 | 610 | 2453 | 0 |
| val | 460 | 460 | 460 | 1953 | 0 |
| test | 610 | 610 | 610 | 3249 | 0 |

## 5. 학습 설정 요약
기준 파일: `runs/obb/runs_yolo_obb/exp01/args.yaml`

| 항목 | 값 |
|---|---|
| task | `obb` |
| model | `yolo11n-obb.yaml` |
| data | `processed_obb/dataset.yaml` |
| epochs | `150` |
| imgsz | `1024` |
| batch | `8` |
| patience | `30` |
| device | `0` (GPU) |
| pretrained | `false` |

## 6. 학습 성능 요약 (최종 epoch)
기준 파일: `runs/obb/runs_yolo_obb/exp01/results.csv` 마지막 행(epoch 150)

| 지표 | 값 |
|---|---:|
| precision(B) | 0.87319 |
| recall(B) | 0.72248 |
| mAP50(B) | 0.84542 | -> 탐지 성능(약 84.5%)
| mAP50-95(B) | 0.63890 |
| train box/cls/dfl/angle loss | 0.77847 / 0.62511 / 1.20347 / 0.01031 |
| val box/cls/dfl/angle loss | 0.91780 / 0.75775 / 1.34508 / 0.01078 |

## 7. 시각화 파일 매칭(의미 정리)

### 7.1 `results/exp01/overview`
| 파일 | 의미 |
|---|---|
| `results.png` | 전체 학습 추세를 요약한 종합 그래프(손실/성능) |

### 7.2 `results/exp01/metrics_curves`
| 파일 | 의미 |
|---|---|
| `BoxPR_curve.png` | Precision-Recall 곡선 |
| `BoxF1_curve.png` | F1 곡선(precision/recall 균형) |
| `BoxP_curve.png` | precision 변화 추세 |
| `BoxR_curve.png` | recall 변화 추세 |

### 7.3 `results/exp01/confusion`
| 파일 | 의미 |
|---|---|
| `confusion_matrix.png` | 혼동행렬(절대 개수 기준) |
| `confusion_matrix_normalized.png` | 혼동행렬(정규화 비율 기준) |

### 7.4 `results/exp01/custom_curves`
| 파일 | 의미 |
|---|---|
| `loss_curve.png` | train/val 손실 곡선(커스텀 저장본) |
| `accuracy_curve.png` | accuracy-like(mAP 계열) 곡선(커스텀 저장본) |

## 8. 추가 참고: 학습 배치 샘플 이미지(원본 runs 폴더)
경로: `runs/obb/runs_yolo_obb/exp01/`

| 파일 | 의미 |
|---|---|
| `train_batch*.jpg` | 학습 배치 입력 샘플(증강 포함) |
| `val_batch*_labels.jpg` | 검증 배치 GT 라벨 시각화 |
| `val_batch*_pred.jpg` | 같은 검증 배치에 대한 모델 예측 결과 |
| `labels.jpg` | 데이터셋 라벨 요약 시각화 |

## 9. 가중치 파일
경로: `runs/obb/runs_yolo_obb/exp01/weights/`

| 파일 | 의미 |
|---|---|
| `best.pt` | 검증 성능 기준 최고 체크포인트 |
| `last.pt` | 마지막 epoch 체크포인트 |
