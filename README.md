# HRSC2016-MS OBB Ship Detection Project

## 1) 프로젝트 개요
- 목표: 해상 선박 탐지를 위해 회전 바운딩 박스(OBB) 기반 데이터셋을 구성하고 YOLO OBB 모델을 학습/평가
- 데이터: HRSC2016-MS (`archive`)
- 클래스: 단일 클래스 `ship`

## 2) 폴더 구조
- `archive/`
  - `AllImages/*.bmp`
  - `Annotations/*.xml`
  - `ImageSets/{train,val,test,trainval}.txt`
- `processed_obb/`
  - `images/{train,val,test}/*.jpg`
  - `labels/{train,val,test}/*.txt` (YOLO OBB 라벨)
  - `dataset.yaml`, `summary.json`
- `runs/obb/runs_yolo_obb/exp01/`
  - 학습 로그, 시각화, 가중치
- `results/exp01/`
  - 중요 PNG 시각화 파일 정리본

## 3) 코드 파일
- `processing_obb.py`
  - HRSC XML(`robndbox`)을 YOLO OBB 라벨로 변환
  - `dataset.yaml` 자동 생성
- `train_yolo_obb.py`
  - OBB 데이터셋 학습
  - epoch 메트릭 추출(`epoch_metrics.json`) 및 그래프 저장
- `eval_test_yolo_obb.py`
  - test split 정량 평가
  - test 이미지 예측 시각화 저장

## 4) 전처리 결과 요약
`processed_obb/summary.json` 기준:
- train: 이미지 610, 객체 2453
- val: 이미지 460, 객체 1953
- test: 이미지 610, 객체 3249
- 누락/손상: 0

## 5) 실행 순서
1. 전처리
```bash
python processing_obb.py
```
2. 학습
```bash
python train_yolo_obb.py
```
3. 테스트 평가/시각화
```bash
python eval_test_yolo_obb.py
```

## 6) 학습 성능 요약
`runs/obb/runs_yolo_obb/exp01/results.csv` 마지막 epoch(150) 기준:
- precision(B): `0.87319`
- recall(B): `0.72248`
- mAP50(B): `0.84542`
- mAP50-95(B): `0.63890`

## 7) runs 폴더 산출물 해설
기준 경로: `runs/obb/runs_yolo_obb/exp01/`

### 7.1 학습 설정/로그 파일
- `args.yaml`
  - 실제 학습에 적용된 하이퍼파라미터 전체 스냅샷
  - 재현 실험 시 기준 파일
- `results.csv`
  - epoch별 수치 로그(손실, P/R, mAP, lr 등)
  - 정량 비교의 원본 테이블
- `epoch_metrics.json`
  - `results.csv`에서 추출한 핵심 곡선 데이터(사용자 정의 저장본)

### 7.2 핵심 시각화 파일
- `results.png`
  - 학습 전반 추세를 요약한 통합 그래프
- `BoxPR_curve.png`
  - Precision-Recall 곡선
  - 임계값 변화에 따른 탐지 성능 비교 핵심
- `BoxF1_curve.png`
  - F1 점수 곡선
  - precision/recall 균형 관점 최적 지점 확인
- `BoxP_curve.png`
  - precision 중심 추세 확인
- `BoxR_curve.png`
  - recall 중심 추세 확인
- `confusion_matrix.png`
  - 절대 개수 기준 오탐/미탐 패턴
- `confusion_matrix_normalized.png`
  - 정규화 비율 기준 오류 패턴
- `loss_curve.png`
  - train/val 손실 곡선(사용자 커스텀)
- `accuracy_curve.png`
  - accuracy-like 지표(mAP 계열) 곡선(사용자 커스텀)

### 7.3 이미지 샘플 파일
- `labels.jpg`
  - 데이터셋 라벨 분포/샘플 확인용 요약 이미지
- `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`, `train_batch10780.jpg` 등
  - 학습 배치 샘플(증강 포함) 시각 확인
- `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg`
  - 검증 배치 GT 라벨 표시 이미지
- `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`
  - 동일 검증 배치의 모델 예측 결과 이미지

### 7.4 가중치 파일
- `weights/best.pt`
  - 검증 성능 기준 최고 체크포인트
- `weights/last.pt`
  - 마지막 epoch 체크포인트

## 8) results 폴더(정리본) 해설
기준 경로: `results/exp01/`

- `overview/`
  - `results.png`
- `metrics_curves/`
  - `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
- `confusion/`
  - `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `custom_curves/`
  - `loss_curve.png`, `accuracy_curve.png`

## 9) Git 업로드 권장
- 커밋 권장: 코드 + README + `.gitignore`
- 제외 권장: `archive/`, `processed_obb/`, `runs/`, `results/`, `*.pt`
