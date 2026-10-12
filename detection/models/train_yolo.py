"""
Detection YOLOv8 Training & Fine-Tuning Pipeline
Trains or fine-tunes YOLOv8n on the Detection VisDrone traffic dataset.
Saves optimized weights to weights/edgeflow_yolov8n.pt.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
from pathlib import Path
from ultralytics import YOLO

DEFAULT_DATA_YAML = Path(r"c:\g\datasets\VisDrone\edgeflow_traffic.yaml")
OUTPUT_WEIGHTS_DIR = Path(r"c:\g\weights")
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"

def train_model(
    data_yaml: Path = DEFAULT_DATA_YAML,
    base_model: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    project_dir: Path = OUTPUT_WEIGHTS_DIR,
    resume: bool = False
):
    last_ckpt = project_dir / "edgeflow_run" / "weights" / "last.pt"

    print(f"=== Detection YOLO Training Pipeline ===")
    print(f"Dataset config: {data_yaml}")
    print(f"Base architecture: {base_model}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Image size: {imgsz}")
    print(f"Execution device: {device}")

    project_dir.mkdir(parents=True, exist_ok=True)

    if resume and last_ckpt.exists():
        print(f"Resuming fine-tuning from existing checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")

        # Initialize YOLO model
        model = YOLO(base_model)

        # Train with traffic-optimized augmentations with batch=8 for 6GB VRAM safety
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=2,
            device=device,
            project=str(project_dir),
            name="edgeflow_run",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.005,
            lrf=0.01,
            mosaic=0.8,
            mixup=0.1,
            save=True,
            val=True,
            plots=True
        )

    best_weight = project_dir / "edgeflow_run" / "weights" / "best.pt"
    target_weight = project_dir / "edgeflow_yolov8n.pt"

    if best_weight.exists():
        import shutil
        shutil.copy(best_weight, target_weight)
        print(f"\nTraining Complete! Best model saved to: {target_weight}")
    else:
        print(f"\nRun completed. Check {project_dir / 'edgeflow_run'} for results.")

    return results

def main():
    parser = argparse.ArgumentParser(description="Detection YOLO Training")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device (0, cpu, etc.)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    args = parser.parse_args()

    train_model(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
