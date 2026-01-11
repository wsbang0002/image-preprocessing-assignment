import os
import random
import numpy as np
import cv2

from datasets import load_dataset

OUT_DIR = "preprocessed_samples"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SIZE = (224, 224)

# --------- Outlier / Filter helpers ----------
def mean_brightness(gray_img: np.ndarray) -> float:
    # gray_img: uint8 [0..255]
    return float(np.mean(gray_img))

def edge_density(gray_img: np.ndarray) -> float:
    """
    간단한 '객체가 너무 작거나 정보가 거의 없는' 이미지 제거용 지표.
    - 실제 '객체 크기' 정답은 없으니, 엣지(윤곽) 비율이 너무 낮으면
      대부분 배경만 있거나 너무 흐리거나 객체가 매우 작은 경우로 보고 제외.
    """
    edges = cv2.Canny(gray_img, 60, 120)
    return float(np.mean(edges > 0))  # 0~1

# --------- Augment helpers ----------
def random_augment(bgr: np.ndarray) -> np.ndarray:
    img = bgr.copy()

    # 좌우 반전 (50%)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # 회전 (-15~+15도)
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 색상 변화 (HSV에서 S/V 약간 조정)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.85, 1.15)  # Saturation
    hsv[..., 2] *= random.uniform(0.85, 1.15)  # Value
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img

def preprocess_one(bgr: np.ndarray) -> dict:
    # 1) Resize
    resized = cv2.resize(bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 2) Blur (noise reduction)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # 3) Grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 4) Normalize (0~1 float)
    norm = gray.astype(np.float32) / 255.0

    return {
        "resized_bgr": resized,
        "blurred_bgr": blurred,
        "gray_u8": gray,
        "normalized_f32": norm,
    }

def save_sample(idx: int, bgr: np.ndarray):
    path = os.path.join(OUT_DIR, f"sample_{idx:02d}.jpg")
    cv2.imwrite(path, bgr)

def main():
    print("Loading dataset (ethz/food101)...")
    ds = load_dataset("ethz/food101", split="train")

    # 필터 기준 (너무 어두운 이미지 제거)
    BRIGHTNESS_MIN = 60.0     # 0~255 기준, 낮을수록 어두움
    EDGE_DENSITY_MIN = 0.01   # 너무 낮으면 정보량 거의 없음

    saved = 0
    tried = 0

    # 랜덤으로 뽑아서 5장 저장
    while saved < 5 and tried < 500:
        tried += 1
        item = ds[random.randrange(0, len(ds))]
        pil_img = item["image"]  # PIL Image

        # PIL -> OpenCV BGR
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 증강 먼저 1번 적용(샘플 다양성)
        aug_bgr = random_augment(bgr)

        # 전처리
        out = preprocess_one(aug_bgr)
        gray = out["gray_u8"]

        # 이상치 제거 1) 너무 어두운 이미지 제거
        mb = mean_brightness(gray)
        if mb < BRIGHTNESS_MIN:
            continue

        # 이상치 제거 2) 객체(정보) 너무 작거나 거의 없는 이미지 제거(엣지 밀도)
        ed = edge_density(gray)
        if ed < EDGE_DENSITY_MIN:
            continue

        # 저장은 보기 좋게 "전처리된 결과"를 저장 (blurred_bgr or gray)
        # 과제는 전처리 결과물이면 OK. 여기선 blurred_bgr 저장.
        save_sample(saved + 1, out["blurred_bgr"])
        saved += 1
        print(f"[{saved}/5] saved | brightness={mb:.1f}, edge_density={ed:.4f}")

    if saved < 5:
        print("Warning: 5장을 못 채웠습니다. 필터 기준을 완화해보세요.")
    else:
        print("Done! Saved 5 samples to preprocessed_samples/")

if __name__ == "__main__":
    main()

