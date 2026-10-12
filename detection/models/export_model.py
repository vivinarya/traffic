"""
Detection Model Exporter & Quantizer
Exports trained YOLOv8n models to Edge formats:
- ONNX (Cross-platform edge runtime)
- TensorRT (.engine for NVIDIA Jetson Nano / Orin)
- OpenVINO / NCNN (for Raspberry Pi 5 / ARM CPUs)
"""
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def export_edge_model(
    model_path: str = "yolov8n.pt",
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False
):
    print(f"Exporting {model_path} to format '{format}'...")
    print(f"Parameters: imgsz={imgsz}, half={half}, int8={int8}")

    model = YOLO(model_path)
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        int8=int8,
        dynamic=False,
        simplify=True
    )
    print(f"\nExport successful! Edge model available at: {export_path}")
    return export_path

def main():
    parser = argparse.ArgumentParser(description="Detection Model Export")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to .pt weights file")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "openvino", "ncnn", "tflite"], help="Target export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    args = parser.parse_args()

    export_edge_model(
        model_path=args.weights,
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8
    )

if __name__ == "__main__":
    main()
