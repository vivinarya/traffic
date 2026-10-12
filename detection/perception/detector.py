"""
Traffic Light Vehicle Density Detector - Perception Layer
Wraps YOLOv8n with dynamic class filtering and Passenger Car Unit (PCU) weighting.
Supports PyTorch (.pt) and exported ONNX models.
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# PCU weights for heterogeneous traffic flow calculation
PCU_WEIGHTS = {
    "car": 1.0,
    "van": 1.5,
    "truck": 3.0,
    "bus": 3.0,
    "motor": 0.5,
    "motorcycle": 0.5,
    "tricycle": 0.8,
    "awning-tricycle": 1.0,
    "bicycle": 0.2,
    "pedestrian": 0.0,
    "person": 0.0,
    "people": 0.0,
    "ambulance": 10.0,
    "emergency": 10.0,
    "police": 10.0,
    "fire truck": 10.0
}

# Standardized vehicle class name mapping
TRAFFIC_NAME_CANONICAL = {
    "car": "car",
    "automobile": "car",
    "van": "van",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "tricycle": "tricycle",
    "auto": "tricycle",
    "awning-tricycle": "awning-tricycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "person": "pedestrian",
    "people": "pedestrian",
    "pedestrian": "pedestrian",
    "ambulance": "ambulance",
    "emergency": "ambulance",
    "police": "police",
    "fire truck": "fire truck"
}

class TrafficLightVehicleDetector:
    """
    Traffic Light Vehicle Density Detector - Detection Engine
    Dynamically identifies vehicle classes from VisDrone or COCO YOLO models,
    applies PCU weights, and supports sensitive detection for small/dark vehicles.
    """
    def __init__(
        self,
        model_path: str = "c:/g/weights/edgeflow_yolov8n.pt",
        conf_threshold: float = 0.18,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        is_visdrone_model: Optional[bool] = None
    ):
        from ultralytics import YOLO
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        print(f"[Traffic Light Vehicle Density Detector] Loading model from {model_path} ({device})...")
        self.model = YOLO(model_path)

        # Determine model domain from names dictionary
        model_names = self.model.names if hasattr(self.model, "names") else {}
        if is_visdrone_model is not None:
            self.is_visdrone_model = is_visdrone_model
        else:
            # Auto-detect: VisDrone typically has 10 classes with 'van' or 'awning-tricycle'
            self.is_visdrone_model = len(model_names) == 10 and any(k in ["van", "awning-tricycle", "tricycle"] for k in model_names.values())

        domain_str = "VisDrone (Aerial/Top-Down)" if self.is_visdrone_model else "COCO/Standard Road Traffic"
        print(f"[Traffic Light Vehicle Density Detector] Auto-detected domain: {domain_str} ({len(model_names)} classes)")

        # Build list of valid class IDs to predict
        self.traffic_class_ids = []
        for cid, name in model_names.items():
            clean_name = str(name).lower().strip()
            if clean_name in TRAFFIC_NAME_CANONICAL:
                self.traffic_class_ids.append(int(cid))

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs object detection on a single frame.
        Returns:
            detections: List of dicts containing bbox [x1, y1, x2, y2], conf, class_id, class_name, pcu_weight
            latency_ms: Inference time in milliseconds
        """
        t0 = time.perf_counter()

        classes_filter = self.traffic_class_ids if self.traffic_class_ids else None

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes_filter,
            device=self.device,
            verbose=False
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        if not results or len(results) == 0:
            return detections, latency_ms

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, latency_ms

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        names = self.model.names

        for i in range(len(xyxy)):
            c_id = int(clss[i])
            raw_name = names.get(c_id, "car").lower().strip()
            canonical_name = TRAFFIC_NAME_CANONICAL.get(raw_name, "car")
            pcu = PCU_WEIGHTS.get(canonical_name, 1.0)

            detections.append({
                "bbox": [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])],
                "confidence": float(confs[i]),
                "class_id": c_id,
                "class_name": canonical_name,
                "pcu_weight": pcu
            })

        return detections, latency_ms

# Backward compatibility alias
EdgeYOLODetector = TrafficLightVehicleDetector
