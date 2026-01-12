import os
import numpy as np
import cv2

def generate_depth_map(image):
    # 사진 지시사항: 입력 None 예외 처리
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    if not isinstance(image, np.ndarray):
        raise TypeError("입력은 numpy.ndarray 여야 합니다.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("입력 이미지는 (H, W, 3) BGR 형식이어야 합니다.")

    # 사진 지시사항: grayscale -> applyColorMap(JET)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return depth_map

def generate_point_cloud_hw3(image):
    """
    사진 지시사항 스타일:
    X,Y = meshgrid
    Z = gray.astype(np.float32)  # Z축
    points_3d = np.dstack((X,Y,Z))
    """
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    if not isinstance(image, np.ndarray):
        raise TypeError("입력은 numpy.ndarray 여야 합니다.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("입력 이미지는 (H, W, 3) BGR 형식이어야 합니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)  # ✅ 사진 강조: Z는 float32

    points_hw3 = np.dstack((X, Y, Z)).astype(np.float32)  # (H,W,3)
    return points_hw3

def flatten_points(points_hw3):
    # (H,W,3) -> (H*W,3)
    if not isinstance(points_hw3, np.ndarray):
        raise TypeError("points_hw3는 numpy.ndarray 여야 합니다.")
    if points_hw3.ndim != 3 or points_hw3.shape[2] != 3:
        raise ValueError("points_hw3는 (H, W, 3) 형태여야 합니다.")
    return points_hw3.reshape(-1, 3).astype(np.float32)

def save_ply(points_n3, filepath):
    # (N,3) ASCII PLY 저장 (Open3D 없이)
    if not isinstance(points_n3, np.ndarray):
        raise TypeError("points_n3는 numpy.ndarray 여야 합니다.")
    if points_n3.ndim != 2 or points_n3.shape[1] != 3:
        raise ValueError("points_n3는 (N, 3) 형태여야 합니다.")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {points_n3.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]) + "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points_n3:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")

def run_demo(sample_path="sample/sample.jpg", out_dir="outputs", show=False):
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(sample_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {sample_path}")

    depth_map = generate_depth_map(image)

    points_hw3 = generate_point_cloud_hw3(image)
    points_n3 = flatten_points(points_hw3)

    cv2.imwrite(os.path.join(out_dir, "depth_map.png"), depth_map)
    save_ply(points_n3, os.path.join(out_dir, "point_cloud.ply"))

    # VM은 show=False 권장 (GUI 없는 경우가 많음)
    if show:
        cv2.imshow("Original Image", image)
        cv2.imshow("Depth Map", depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo(show=False)
