# 문서 한눈에 MVP

공공문서 이미지를 품질 검사한 뒤 OCR하고, Gemini가 원문 줄 번호를 근거로 핵심 내용과 해야 할 일을 요약합니다. AI 호출이 실패하면 출처가 유지되는 로컬 발췌 요약으로 완료합니다.

## 현재 기능

- JPEG/PNG 업로드와 촬영 품질 판정
- 1536px PaddleOCR PP-OCRv5 한국어 문서 OCR과 적응형 Tesseract 장애 폴백
- Gemini 구조화 요약과 원문 줄 단위 출처 연결
- 출처 버튼을 통한 이미지 OCR 영역 하이라이트
- 요약 정정 피드백 SQLite 저장
- 고해상도 원본·Paddle 문서 펴기·일반 OCR 기준선 비교 화면
- AI-Hub Training→train/dev, Validation→final 전용 manifest 생성
- 좌표 기반 범용 읽기 순서 정렬
- 분리된 공공문서 OCR GPU 학습·PaddleOCR-VL 비교 도구

## 실행

`.env`에는 `GEMINI_API_KEY`를 둡니다. 현재 파일의 `AI_API_KEY` 이름도 호환합니다. 이 파일은 Docker 이미지와 Git 대상에서 제외됩니다.

```powershell
docker build -t kcode-public-doc-mvp .
docker run --rm --env-file .env -p 8000:8000 kcode-public-doc-mvp
```

- 실제 사용 화면: `http://localhost:8000/`
- 비교 실험 화면: `http://localhost:8000/test`
- API 문서: `http://localhost:8000/docs`

기본 모델은 실제 구조화 요약과 무료 할당량 경로를 확인한 `gemini-3.5-flash-lite`입니다. 필요하면 `.env`의 `GEMINI_MODEL`로 바꿀 수 있습니다.

## 비교 실험의 의미

| 구분 | 고해상도 원본 | 문서 펴기 실험군 | 일반 기준선 |
|---|---|---|---|
| 이미지 | 품질 게이트, 원본 1536px 보존 | Paddle UVDoc 언워핑 | 원본 회색조 |
| OCR | PP-OCRv5 한국어 모델 | 같은 PP-OCRv5 한국어 모델 | 배포판 기본 Tesseract 자동 페이지 모드 |
| 요약 | 구조화 출력, 줄 인용 강제, 서버 검증 | 실행하지 않음 | 일반 요약 프롬프트 |
| 목적 | 실제 사용 기본값 | 필터 ON/OFF 분리 비교 | 기존 방식 비교 |

화면의 단일 실행 시간과 모델 확신도는 연구 성능의 최종 결론이 아닙니다. OCR 성능은 AI-Hub의 순차 CER/WER와 실제 스마트폰 촬영본의 순서 독립 단어 F1·숫자 F1을 함께 봅니다. 표와 다단 문서는 읽기 순서만 달라도 CER이 크게 오를 수 있습니다.
AI-Hub 라벨이 화면의 일부 텍스트만 포함해 정상 인식분을 삽입 오류로 계산하는 경우가 있으므로, 출력이 정답의 2배 이상이면서 CER이 1을 넘는 문서는 `alignment_review_required`로 자동 표시하고 원본·라벨을 함께 검토합니다.

Inv3DReal 실제 스마트폰 사진 48장에서는 UVDoc이 원본보다 단어 F1 +3.17%p, 숫자 F1 +2.95%p 높았습니다. 다만 11장에서는 악화됐고 처리 시간이 6.24초에서 8.89초로 늘며 원본 이미지 하이라이트 좌표도 직접 유지할 수 없습니다. 그래서 실제 사용 화면은 안정적인 원본 1536px 경로를 유지하고, `/test`에서 UVDoc 결과를 함께 확인하도록 구성했습니다.

CPU 호환성 문제를 피하기 위해 PaddlePaddle 3.3.1의 MKLDNN은 끕니다. `OCR_ENGINE=tesseract`로 두면 장애 폴백 경로만 단독 실행할 수 있습니다.

공공문서 인식기 미세조정, GPU clone 실행, 중단 시 자동 체크포인트·재개, PaddleOCR-VL-1.5/1.6 비교 절차는 [ocr_training/README.md](ocr_training/README.md)에 분리했습니다. 검증을 통과한 exported 모델은 `PADDLE_RECOGNITION_MODEL_DIR`로 연결하며, 설정하지 않으면 기존 공식 한국어 PP-OCRv5 모델을 그대로 사용합니다.

## 검사

```powershell
docker run --rm kcode-public-doc-mvp pytest
docker run --rm kcode-public-doc-mvp ruff check app research tests
docker run --rm kcode-public-doc-mvp mypy app research tests
```

AI-Hub OCR 비교 평가는 원본 ZIP을 읽기 전용으로 연결해 실행합니다.

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -v "${PWD}/Test Data:/app/Test Data:ro" `
  kcode-public-doc-mvp `
  python -m research.scripts.evaluate_ocr --split smoke `
    --profile baseline --profile paddle `
    --output research/results/ocr_paddle_smoke.json
```

실제 스마트폰 촬영본의 필터 비교는 Inv3DReal ZIP을 풀지 않고 실행합니다. 데이터 출처, 체크섬, 실험 결과와 채택 근거는 [research/PHOTO_EVALUATION.md](research/PHOTO_EVALUATION.md)에 기록했습니다.

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  kcode-public-doc-mvp `
  python -m research.scripts.evaluate_photo_filters `
    --archive-limit 2 --templates-per-archive 4 `
    --variant raw --variant paddle_unwarp `
    --det-limit 1536 --output research/results/photo_filter_inv3d_48.json
```

## AI-Hub manifest 생성

원본 ZIP은 읽기 전용으로 연결하고 압축을 풀지 않습니다.

```powershell
docker run --rm `
  -v "${PWD}/Test Data:/input:ro" `
  -v "${PWD}/research/manifests:/output" `
  kcode-public-doc-mvp `
  python -m research.scripts.build_manifests --test-data /input --output /output
```

## 데이터 출처와 공개 범위

이 프로젝트의 공공문서 OCR 학습과 평가는 과학기술정보통신부와 한국지능정보사회진흥원(NIA)의 AI-Hub 사업 결과인 `공공행정문서 OCR 데이터`를 활용합니다. 데이터 사용 시 [AI-Hub 데이터 이용정책](https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)을 따릅니다.

AI-Hub 원본 이미지, 정답 라벨, 원문이나 OCR 결과가 포함된 상세 평가 JSON은 이 저장소에 공개하지 않습니다. 데이터는 이용자가 AI-Hub에서 직접 사용 신청하고 내려받아야 하며, 저장소에는 재현을 위한 코드와 원문을 포함하지 않은 설명만 공개합니다.
