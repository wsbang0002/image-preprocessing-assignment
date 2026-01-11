# Image Preprocessing Assignment

## 1. 개요
Hugging Face `ethz/food101` 데이터셋에서 이미지를 불러와 AI 학습용 전처리를 수행했습니다.

## 2. 수행 내용
- Resize: 224x224
- Noise Removal: Gaussian Blur
- Color Transform: Grayscale
- Normalize: 0~1 스케일
- Data Augmentation: 좌우반전, 회전(-15~+15도), 색상(S/V) 변화
- Outlier Filtering:
  - 너무 어두운 이미지 제거 (평균 밝기 기준)
  - 정보량이 너무 적은 이미지 제거 (엣지 밀도 기준)

## 3. 실행 방법
```bash
pip install opencv-python numpy pillow datasets
python image_preprocessing.py

