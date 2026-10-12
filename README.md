# EdgeFlow

Lightweight vision-based framework for distributed predictive traffic signal control with embedded physical implementation.

EdgeFlow integrates low-latency YOLO object detection, ByteTrack multi-object tracking, and queue kinematics with an adaptive signal controller and physical microcontroller actuation. It predicts short-horizon traffic arrivals, calculates optimal phase allocations under safety constraints, and streams real-time phase decisions to an Arduino-interlocked traffic signal rig over USB serial.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Execution Flow](#execution-flow)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Interactive Lane Calibration](#2-interactive-lane-calibration)
  - [3. Run Application](#3-run-application)
- [Usage Example](#usage-example)
  - [Perception Pipeline and Adaptive Controller](#perception-pipeline-and-adaptive-controller)
  - [Hardware Bridge Diagnostic](#hardware-bridge-diagnostic)
- [Testing](#testing)

---

## Key Features

- **High-Throughput Edge Perception**: YOLOv8n object detection optimized for heterogeneous traffic categories (cars, buses, trucks, vans, motorcycles) with Passenger Car Unit (PCU) weighting.
- **Robust Multi-Object Tracking**: ByteTrack trajectory estimation for velocity tracking, approach headway computation, and stop-line arrival verification.
- **Spatial Kinematics and Queue Estimation**: Polygonal lane geometry and stop-line boundary checks for real-time queue length, vehicle accumulation, and arrival rate calculations.
- **Predictive Adaptive Controller**: Demand-responsive control law balancing local queue pressure, short-term arrival forecasts, and neighbor intersection state under min/max green and yellow-clearance constraints.
- **Embedded Hardware Interlocking**: USB serial bridge communicating with Arduino firmware enforcing hardware-level phase separation, deterministic timing, and watchdog fallback.
- **Thread-Safe Telemetry Ledger**: SQLite database with Write-Ahead Logging (WAL) recording per-vehicle kinematic events, lane aggregates, and controller phase transitions.

---

## Architecture Overview

```text
edgeflow/
├── arduino_traffic_lights.ino      # Physical Arduino signal controller firmware
├── calibrate_lanes.py              # Interactive OpenCV ROI and stop-line calibration
├── lane_calibration.json           # Calibrated intersection approach geometry
├── run_detector.bat                # Windows execution launcher
├── test_arduino.py                 # Serial hardware diagnostic and LED cycle test
├── traffic_light_detector_app.py   # Master application and live HUD orchestrator
├── detection/                      # Core detection and control engine
│   ├── controller/                 # Traffic state, predictor, and adaptive controller
│   │   ├── adaptive_controller.py  # Phase selection and green duration control law
│   │   ├── arduino_bridge.py       # Serial protocol bridge for hardware communication
│   │   ├── short_term_predictor.py # Rolling arrival rate and demand forecasting
│   │   ├── signal_simulator.py     # Software traffic light state machine
│   │   └── traffic_state.py        # Approach state representation and metrics
│   ├── database/                   # SQLite persistence and audit telemetry
│   │   ├── db_manager.py           # Thread-safe connection and query manager
│   │   └── schema.sql              # Database schema definitions (WAL mode)
│   ├── models/                     # Model training and export utilities
│   │   ├── export_model.py         # ONNX / TensorRT export utility
│   │   └── train_yolo.py           # YOLO fine-tuning pipeline
│   ├── perception/                 # Computer vision perception modules
│   │   ├── detector.py             # YOLO inference wrapper and PCU weighting
│   │   ├── kinematics.py           # Speed, headway, and queue estimation
│   │   ├── lane_geometry.py        # Lane polygon containment and stop-line checks
│   │   └── tracker.py              # ByteTrack trajectory management
│   └── pipeline.py                 # Integrated video processing pipeline
├── tests/                          # Integration and validation test suite
│   ├── benchmark_edge.py           # Edge CPU/GPU inference latency benchmark
│   ├── generate_test_traffic.py    # Synthetic traffic video generator
│   ├── prepare_real_traffic_video.py # Video preprocessing helper
│   ├── test_gapout_and_preemption.py # Gap-out and emergency preemption tests
│   ├── test_pipeline.py            # End-to-end perception and DB integration test
│   └── test_traffic_controller.py  # Controller decision and phase logic tests
└── weights/                        # Trained neural network weights
```

---

## Execution Flow

1. **Frame Ingestion**: Video feeds (or live RTSP/webcam inputs) are decoded and ingested frame-by-frame.
2. **Inference & Association**: The YOLO detector identifies vehicle bounding boxes, and ByteTrack associates trajectories to track movement history and speeds.
3. **Kinematic Projection**: Tracked objects are mapped onto calibrated lane polygons to compute approach density, stop-line queues, and arrival rates.
4. **State Estimation & Forecasting**: The traffic state manager aggregates queue pressures, while the short-term predictor estimates incoming vehicle volumes.
5. **Adaptive Phase Allocation**: The controller applies green duration bounds, evaluates queue disparities between competing approaches, and schedules phase switches.
6. **Actuation & Telemetry**: Signal directives are dispatched to the Arduino firmware over USB serial, while state metrics are committed to the telemetry database.

---

## Prerequisites

- **Python**: 3.10 or higher
- **PyTorch**: Compatible with CUDA for accelerated edge inference (or CPU execution)
- **Dependencies**: `ultralytics`, `opencv-python`, `pyserial`, `numpy`, `scipy`
- **Embedded Hardware (Optional)**: Arduino Uno / Nano with 6 LEDs (Red, Yellow, Green for North-South and East-West approaches)

---

## Quickstart

### 1. Environment Setup

Clone the repository and install required Python packages:

```bash
# Clone the repository
git clone https://github.com/your-org/edgeflow.git
cd edgeflow

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install ultralytics opencv-python pyserial numpy scipy pytest
```

### 2. Interactive Lane Calibration

Calibrate lane polygons and stop lines using the interactive OpenCV tool:

```bash
python calibrate_lanes.py --video tests/real_traffic_video.mp4
```

Click to define corner points for each approach polygon and stop line. Calibration parameters are saved to `lane_calibration.json`.

### 3. Run Application

Run the master detection and control application:

```bash
# Run with simulated software signal HUD
python traffic_light_detector_app.py --source tests/real_traffic_video.mp4

# Run with physical Arduino hardware connected (e.g. COM3)
python traffic_light_detector_app.py --source tests/real_traffic_video.mp4 --arduino COM3
```

---

## Usage Example

### Perception Pipeline and Adaptive Controller

```python
from pathlib import Path
from detection.pipeline import EdgeFlowPerceptionPipeline
from detection.controller.adaptive_controller import AdaptiveTrafficSignalController, SignalPhase

# Initialize perception pipeline
pipeline = EdgeFlowPerceptionPipeline(
    model_weights="weights/detection_yolov8n.pt",
    db_path=Path("telemetry.db"),
    conf_thresh=0.25,
    device="0"
)

# Initialize adaptive signal controller
controller = AdaptiveTrafficSignalController(
    min_green_s=10.0,
    max_green_s=45.0,
    yellow_s=3.0,
    all_red_s=1.0
)

# Process video and receive operational metrics
summary = pipeline.process_video(
    video_source="tests/real_traffic_video.mp4",
    output_path="annotated_output.mp4",
    max_frames=100
)

print(f"Processed {summary['frames_processed']} frames @ {summary['average_fps']:.1f} FPS")
```

### Hardware Bridge Diagnostic

Verify serial communication and physical LED switching on the microcontroller:

```python
from detection.controller.arduino_bridge import ArduinoTrafficLightBridge
from detection.controller.adaptive_controller import SignalPhase

# Connect to Arduino over USB serial
bridge = ArduinoTrafficLightBridge(port="COM3", baud_rate=115200)
bridge.connect()

# Set North-South approach Green, East-West Red
bridge.send_phase(SignalPhase.NS_GREEN)

# Close connection
bridge.disconnect()
```

Alternatively, run the interactive hardware test utility:

```bash
python test_arduino.py
```

---

## Testing

Execute the test suite to validate database persistence, kinematics, phase logic, and edge throughput:

```bash
# Run all unit and integration tests
pytest tests

# Run edge inference benchmark
python tests/benchmark_edge.py
```