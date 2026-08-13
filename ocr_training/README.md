# 공공문서 OCR GPU 학습

AI-Hub 공공행정문서 OCR의 전체 Training 데이터로 한국어 PP-OCRv5 글자 인식 모델을 한 번에 미세조정합니다. 원본 ZIP은 읽기 전용으로 사용하며 학습용 글자 이미지를 디스크에 만들지 않습니다.

## 준비물

- NVIDIA GPU, Docker Engine, Docker Compose
- 이미 내려받은 AI-Hub Training/Validation 전체 데이터
- 이미 내려받은 `korean_PP-OCRv5_mobile_rec_pretrained.pdparams`

호스트에 Python, CUDA 패키지 또는 PaddleOCR을 설치하지 않습니다.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <CLONED_REPOSITORY>
cp .env.ocr.example .env.ocr
```

`.env.ocr`에서 기존 파일 위치를 지정합니다.

```dotenv
AIHUB_OCR_DATA=/data/공공행정문서 OCR
OCR_PRETRAINED_MODEL=ocr_training/artifacts/pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams
OCR_BASELINE_MODEL_DIR=
OCR_EPOCHS=20
OCR_BATCH_SIZE=64
OCR_SAVE_BATCH_STEP=500
```

`OCR_PRETRAINED_MODEL`은 컨테이너의 `/workspace`, 즉 clone한 저장소를 기준으로 한 경로입니다. 파일이 없으면 자동 다운로드하지 않고 중단합니다.

## 전체 실행

```bash
bash ocr_training/run_gpu.sh --stage all
```

한 명령이 다음을 순서대로 수행합니다.

1. GPU, 기존 데이터, 기존 사전학습 모델 검사
2. Training 90% train, 10% dev와 Validation final 문서 목록 작성
3. 작은 SQLite·NumPy 인덱스 생성
4. 전체 Training 데이터를 대상으로 단일 학습 또는 이전 중단 지점 재개
5. 최적 체크포인트를 앱용 추론 모델로 내보내기
6. AI-Hub Validation 표본으로 학습 모델 평가
7. 현재 PC로 옮길 `public_doc_rec.zip`과 SHA-256 생성

소규모 데이터로 먼저 가중치를 학습하는 별도의 1차 과정은 없습니다. 실행 전 검사는 파일을 읽을 수 있는지만 확인하며 모델 가중치를 바꾸지 않습니다.

기존 추론 모델도 이미 가지고 있다면 `OCR_BASELINE_MODEL_DIR`에 그 폴더를 지정해 같은 표본으로 추가 비교할 수 있습니다. 비워 두면 새 모델을 자동으로 내려받지 않고 학습 모델만 평가합니다.

## 디스크를 적게 쓰는 방식

`prepare`는 `images/train`이나 `images/dev`를 만들지 않습니다. 문서 위치와 유효한 글자 위치만 저장합니다. 학습 worker는 ZIP에서 한 페이지를 읽고 RAM에서 글자 영역을 crop한 뒤 GPU로 보내며, 제한된 페이지 캐시에서 밀려난 데이터는 즉시 버립니다.

AI-Hub Validation의 최종 비교 표본만 clean/camera/camera_filtered 세 형태로 저장합니다. 기본 120문서이므로 크기가 제한됩니다.

## 중단과 재개

- 기본 500 batch마다 `recovery_step_N` 저장
- 매 epoch `latest` 저장
- 가장 정확한 개발 평가 모델은 `best_accuracy`로 저장
- `Ctrl+C`, SIGTERM 또는 `docker stop`은 현재 batch 뒤 저장하고 종료
- 같은 명령을 다시 실행하면 마지막 완전한 checkpoint에서 재개
- 데이터 인덱스가 달라지면 오래된 checkpoint 재사용을 거부
- 갑작스러운 전원 차단으로 쓰다 만 파일은 `.complete` 표식이 없어 무시

학습이 모든 epoch를 끝내기 전에는 `training_complete.json`이 만들어지지 않으므로 export와 패키징도 시작되지 않습니다.

## 산출물

```text
ocr_training/work/manifests/            문서 목록
ocr_training/data/documents.sqlite3     문서 위치 인덱스
ocr_training/data/*_sample_ends.npy     글자 번호 인덱스
ocr_training/data/images/final/         제한된 최종 평가 표본
ocr_training/artifacts/checkpoints/     재개용 checkpoint
ocr_training/artifacts/inference/       앱용 추론 모델
ocr_training/artifacts/evaluation/      비교 결과 JSON
ocr_training/artifacts/deliverables/
├─ public_doc_rec.zip
└─ public_doc_rec.zip.sha256
```

GPU 컴퓨터에서 `public_doc_rec.zip`과 `.sha256` 두 파일을 현재 PC로 복사합니다. ZIP을 푼 뒤 앱 환경 변수에 `model` 폴더를 지정합니다.

```dotenv
PADDLE_RECOGNITION_MODEL_DIR=C:\models\public_doc_rec\model
```

ZIP에는 추론 모델, 학습 데이터 지문, Git commit, 평가 요약, 내부 파일 체크섬과 한국어 설치 설명이 포함됩니다. 원본 AI-Hub 글자나 상세 평가 결과는 넣지 않습니다.

## 단계별 재실행

```bash
bash ocr_training/run_gpu.sh --stage prepare
bash ocr_training/run_gpu.sh --stage train
bash ocr_training/run_gpu.sh --stage export
bash ocr_training/run_gpu.sh --stage evaluate
bash ocr_training/run_gpu.sh --stage package
```

준비 설정을 바꾼 경우에만 `--force-prepare`를 사용합니다. 기존 crop 방식의 `train.txt`, `dev.txt`, `images/train`, `images/dev`는 새 prepare가 정리합니다.

## GitHub에 포함하지 않는 것

`.gitignore`와 `.dockerignore`는 `.env.ocr`, AI-Hub 원본, manifest, 인덱스, 사전학습 모델, checkpoint, 추론 모델과 평가 결과를 제외합니다. AI-Hub 데이터는 이용 약관에 따라 GPU 컴퓨터에 별도로 보관하십시오.
