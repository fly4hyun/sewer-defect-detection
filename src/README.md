# 서울디지털재단 하수관로 결함 탐지 프로젝트

하수관로 **이미지·영상**에서 크랙·변형 등 결함을 자동 검출·분류하는 **YOLO 기반 AI 파이프라인**입니다.  
이미지/영상 API, 모델 학습·평가 스크립트, (Docker 실행 환경까지 포함되어 있습니다.)
Docker 이미지는 이준영 연구원이 보유
!!! 현재 버전은 하수관로 과제 기간동안 진행한 소스코드로, 이후 연계나 연동 이후에 개선된 버전은 이준영 연구원이 보유 !!!

---

## 1. 디렉터리 구조
```
├── classifier_weights/          # 후처리 분류 모델 가중치
├── data/                        # 원천 데이터 + train/val/test 목록(txt)
├── datasets/                    # YOLO 학습·평가용 데이터셋
├── results/                     # 모델 평가 결과
├── utils/                       # 공용 유틸리티
├── video/                       # 분석 대상 원본 영상
├── video_test_results/          # 분석 완료 영상(mp4)
├── video_test_image_results/    # 분석 완료 영상의 프레임별 이미지
├── results_video/               # 기타 결과 영상
├── weight/                      # 학습된 YOLO 가중치 (.pt)

# 주요 스크립트
├── sewer_simulation_api.py      # 단일 이미지 API
├── track_api_new.py             # 영상 API
├── train.py                     # 모델 학습
├── test.py                      # 모델 평가
├── track.py                     # 영상 실시간 추론
├── yolo_data_generate.py        # 원천 → YOLO 형식 변환

# 실행/배포
├── sewer_api.tar                # Docker 이미지 (API 서버) -> 이준영 연구원이 보유
├── simulation_api_run.sh        # 이미지 API 실행
└── track_api_run.sh             # 영상 API 실행
```

---

## 2. 주요 파일 설명

### 2‑1. Python 스크립트

| 파일 | 기능 |
|------|------|
| `sewer_simulation_api.py` | 단일 **이미지** 입력 → 결함 탐지 결과(JSON) 반환 |
| `track_api_new.py` | **영상** 입력 → 프레임별 결함 탐지 결과(JSON) 반환 |
| `train.py` | YOLO 모델 학습 |
| `test.py` | 학습 모델 성능 평가 (mAP 등) |
| `track.py` | 영상 실시간 추론 & 시각화 (`--save-vid` 지원) |
| `yolo_data_generate.py` | 원천 데이터를 YOLO 학습 형식(이미지+txt)으로 변환 |

### 2‑2. 셸 스크립트

| 파일 | 기능 |
|------|------|
| `simulation_api_run.sh` | 이미지 API 서버 실행 |
| `track_api_run.sh` | 영상 API 서버 실행 |

### 2‑3. 모델 가중치

| 파일 | 학습 범위 |
|------|-----------|
| `YOLOv10x.pt` | 대표 결함만 학습 |
| `YOLOv10x_all.pt` | 대표 + 기타 결함 전체 |
| `YOLOv10x_etc.pt` | 기타 결함 전용 |

---

## 3. 빠른 시작

### 3‑1. 의존성 설치
```bash
conda create -n pipe_demo python=3.12 -y
conda activate pipe_demo
pip install -r requirements.txt
```

### 3‑2. Docker API 실행 -> 해당 부분은 이준영 연구원에게 문의
```bash
# Docker 이미지 로드
docker load -i sewer_api.tar

# 이미지 API  (http://localhost:8000/predict/image)
bash simulation_api_run.sh

# 영상 API   (http://localhost:8000/predict/video)
bash track_api_run.sh
```

### 3‑3. 학습 · 평가 · 추론
```bash
# 학습
python train.py
# 평가
python test.py
# 영상 추론
python track.py
```
> 결과 영상은 `video_test_results/`, 프레임 이미지는 `video_test_image_results/`에 저장됩니다.

---

## 4. 데이터 준비
```bash
데이터셋은 YOLO 학습용으로 상위 폴데 내 1.2.데이터 폴더 내 학습용데이터셋 폴더에 존재
data 폴더 내 txt 파일은 선별된 train과 test 이미지 목록
```

---

## 5. 라이선스 & 문의

> 본 프로젝트는 서울디지털재단 연구 과제로 개발되었습니다.  
> 재배포·2차 활용 시 재단 가이드라인을 준수해 주세요.
