import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="task4/outputs/train/weights/best.pt")
    ap.add_argument("--image", default="datasets/human_eyes/test/images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save_dir", default="task4/outputs/infer")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # image 인자가 폴더면 폴더 내 첫 jpg/png를 하나 선택
    img_path = Path(args.image)
    if img_path.is_dir():
        cands = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")) + list(img_path.glob("*.jpeg"))
        if not cands:
            raise FileNotFoundError(f"No images in {img_path}")
        img_path = cands[0]

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    results = model.predict(source=img, conf=args.conf, verbose=False)
    annotated = results[0].plot()  # bbox가 그려진 이미지 (numpy)

    out_path = save_dir / f"pred_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"[Saved] {out_path}")

    if args.show:
        cv2.imshow("prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
