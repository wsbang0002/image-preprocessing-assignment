import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/human_eyes/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")  # GPU 있으면 0 으로
    ap.add_argument("--project", default="task4/outputs")
    ap.add_argument("--name", default="train")
    args = ap.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Train
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
    )

    # Val (정량 지표 출력/저장)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
    )
    print("\n=== VAL METRICS ===")
    print(metrics)

if __name__ == "__main__":
    main()
