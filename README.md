# Capstone-style CV Project (Week 4) - Drowsiness Eye Detection (Open/Closed)

## 0) Summary (Week 4 제출용)
- **Goal**: YOLOv8 기반으로 사람 눈 상태(**open / closed eye**)를 탐지하여 졸음운전(눈 감김) 위험 신호를 판단할 수 있는 기반 모델/파이프라인을 구축합니다.
- **Key Outputs**
  - 학습 스크립트: `task4_train.py`
  - 추론 스크립트: `task4_infer.py`
  - 추론 결과(전/후 비교용 이미지):
    - `task4/outputs/infer/original.jpg`  (원본)
    - `task4/outputs/infer/pred_*.jpg`   (탐지 결과 시각화)
- **Git Workflow**: `feature/task4-drowsy-eye` 브랜치에서 개발 → PR 생성 → 리뷰/머지

---

## 1) Repository Structure
> 데이터셋/대용량 파일은 GitHub에 올리지 않고(권장), 코드/결과 이미지/문서만 버전관리합니다.

- `task4_train.py` : YOLOv8 학습 스크립트
- `task4_infer.py` : 단일 이미지 또는 폴더 추론 스크립트
- `task4/outputs/infer/` : 추론 결과 저장 폴더
- `datasets/human_eyes/` : (로컬) 데이터셋 위치 (GitHub에는 미포함 가능)

---

## 2) Environment Setup (Ubuntu 22.04 기준)
### (1) venv 생성 (권장)
```bash
python3 -m venv .venv
source .venv/bin/activate

