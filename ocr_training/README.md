# 공공문서 OCR GPU 학습

AI-Hub 공공행정문서 OCR로 `korean_PP-OCRv5_mobile_rec`을 미세조정하는 독립 패키지입니다. 저장소에는 AI-Hub 원본, 생성 crop, 체크포인트를 넣지 않습니다. GPU 머신에서 저장소를 clone하고 AI-Hub 데이터를 읽기 전용으로 연결해 실행합니다.

## 데이터 계약

실제 로컬 AI-Hub Validation을 읽어 다음 구조를 확인했습니다.

```text
공공행정문서 OCR/
└─ Validation/
   ├─ [원천]validation.zip
   │  └─ 02.원천데이터(Jpg)/업무유형/기관/연도/*.jpg
   └─ [라벨]validation.zip
      └─ 01.라벨링데이터(Json)/업무유형/기관/연도/*.json
```

JSON은 `images[0]["image.file.name"]`, `images[0]["image.category"]`, `annotations[*]["annotation.text"]`, `annotations[*]["annotation.bbox"]`를 사용합니다. Training이 여러 ZIP 조각으로 나뉘거나 파일명이 달라도 ZIP 내부의 JPG/PNG와 JSON을 검사해 자동 구분하고, JSON에 선언된 이미지 파일명으로 짝을 맞춥니다. ZIP을 풀거나 수정하지 않습니다.

분할은 코드가 다음과 같이 강제합니다.

- AI-Hub `Training`: 업무 유형별 문서 단위 90% train + 10% dev
- AI-Hub `Validation`: final 전용, 학습·튜닝 경로 진입 금지
- 같은 문서의 모든 텍스트 crop과 촬영 열화본: 항상 같은 분할
- train: clean 55% + 촬영 열화 45%; dev: clean; final: clean/camera/filtered 모두 비교

## GPU 머신에서 바로 실행

요구 사항은 NVIDIA Driver, Docker Engine, Docker Compose와 충분한 디스크입니다. 호스트 Python이나 CUDA 패키지는 설치하지 않습니다. 이미지가 PaddlePaddle 3.3.1 GPU, CUDA 12.6, PaddleOCR v3.7.0 환경을 제공합니다.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <CLONED_REPOSITORY>
cp .env.ocr.example .env.ocr
```

`.env.ocr`의 경로만 실제 다운로드 위치로 바꿉니다.

```dotenv
AIHUB_OCR_DATA=/data/공공행정문서 OCR
OCR_EPOCHS=20
OCR_BATCH_SIZE=64
OCR_SAVE_BATCH_STEP=500
```

데이터 경로에는 `Training`과 `Validation`이 모두 있어야 합니다. 최초 전체 실행은 다음 한 줄입니다.

```bash
bash ocr_training/run_gpu.sh --stage all
```

첫 Docker build는 약 15GB급 CUDA/Paddle 베이스 계층을 내려받을 수 있어 네트워크와 디스크에 따라 오래 걸립니다. 이 명령은 환경 검사 → manifest → recognition crop → 사전학습 모델 → 학습 → export → final 전후 평가를 순서대로 수행합니다. 데이터 생성이 끝난 뒤 재실행하면 완료된 준비 단계를 건너뜁니다. 준비 결과를 다시 만들 때만 `--force-prepare`를 붙입니다.

빠른 end-to-end 확인은 실제 데이터 중 문서 수만 제한합니다.

```bash
bash ocr_training/run_gpu.sh --stage prepare \
  --max-training-documents 120 --final-documents 24
bash ocr_training/run_gpu.sh --stage train --epochs 1 --batch-size 16 \
  --save-batch-step 20
```

## 중단·일시정지·자동 재개

학습 산출물은 clone된 저장소의 `ocr_training/artifacts/checkpoints`에 지속 저장됩니다.

- 매 epoch: `latest`와 `iter_epoch_N` 전체 체크포인트 저장
- 기본 500 batch마다: `recovery_step_N` 저장, 최근 완료본 3개 유지
- `Ctrl+C`, `docker stop`, SIGTERM: 현재 batch를 마친 뒤 recovery 저장 후 종료
- 재실행: 가장 진행도가 높은 완전한 체크포인트를 자동 선택
- 복구 범위: 모델 가중치, optimizer, epoch, global step, 현재 epoch의 다음 batch
- 갑작스러운 전원 차단/OOM/SIGKILL: 쓰다 만 generation은 `.complete` 표식이 없어 무시하고 직전 완료본에서 재개

`latest`, `best_accuracy`, epoch 저장본도 동일한 완료 표식을 사용합니다. export는 완전한 `best_accuracy`를 우선하고, 짧은 시험 학습처럼 best가 아직 없을 때만 최신 완전 체크포인트를 사용합니다.

일시정지는 터미널에서 `Ctrl+C`를 한 번 누르고 `graceful stop completed` 로그를 확인합니다. Docker Compose의 종료 유예 시간은 10분입니다. 즉시 강제 종료하면 마지막 주기 저장 이후 batch는 다시 실행될 수 있습니다.

```bash
# 같은 명령을 다시 실행하면 자동 재개합니다.
bash ocr_training/run_gpu.sh --stage train

# 의도적으로 처음부터 시작할 때만 사용합니다.
docker compose --env-file .env.ocr -f compose.ocr-gpu.yaml run --rm ocr-trainer \
  python -m ocr_training.scripts.train --no-resume
```

batch 저장 주기는 디스크 쓰기와 손실 허용량의 절충입니다. 학습이 불안정한 환경은 `OCR_SAVE_BATCH_STEP=100`, 안정적인 환경은 500을 권장합니다.

## 단계별 실행과 결과

```bash
bash ocr_training/run_gpu.sh --stage prepare
bash ocr_training/run_gpu.sh --stage train
bash ocr_training/run_gpu.sh --stage export
bash ocr_training/run_gpu.sh --stage evaluate
```

주요 로컬 산출물은 모두 Git에서 제외됩니다.

```text
ocr_training/work/manifests/       Training/Validation 문서 계약
ocr_training/data/                 train/dev/final recognition crop
ocr_training/artifacts/checkpoints 전체 재개 체크포인트
ocr_training/artifacts/inference/  앱 연결용 exported model
ocr_training/artifacts/evaluation/ 공식 모델과 미세조정 모델 비교 JSON
```

final 평가는 AI-Hub Validation에서 만든 동일한 clean/camera/filtered 목록으로 공식 모델과 미세조정 모델을 비교합니다. 앱 채택 전에는 clean CER이 악화되지 않고 camera CER이 개선되는지, 실제 스마트폰 사진의 단어·숫자 F1, 전체 문서 처리 시간까지 별도로 확인합니다.

앱에서 채택한 모델은 다음 환경 변수로만 연결합니다.

```text
PADDLE_RECOGNITION_MODEL_DIR=/models/public_doc_rec
```

## 저장소에 포함하지 않는 것

`.gitignore`와 `.dockerignore`는 다음을 제외합니다.

- `.env`, `.env.ocr`, API key
- `Test Data`, AI-Hub 원본 ZIP
- 생성 manifest/crop
- 사전학습·체크포인트·export 모델

GitHub에는 소스와 재현 설정만 올리고, GPU 머신에서 AI-Hub 이용 약관에 따라 데이터를 별도로 내려받으십시오.
