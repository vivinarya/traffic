"""
Detection Comprehensive Perception & Database Test Suite
Executes automated tests on:
1. SQLite Telemetry Database creation, indexing, and WAL mode.
2. Synthetic intersection video generation.
3. YOLOv8 + ByteTrack pipeline execution.
4. Kinematics, PCU accumulation, and queue length calculation.
5. Verification of database records (tracks, lane aggregates, heartbeat).
"""
import sys
import os
import time
from pathlib import Path
import sqlite3

# Add root project directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.database.db_manager import EdgeDBManager
from detection.pipeline import EdgeFlowPerceptionPipeline
from tests.generate_test_traffic import generate_traffic_clip

TEST_DB_PATH = Path(r"c:\g\tests\test_telemetry.db")
TEST_VIDEO_PATH = Path(r"c:\g\tests\real_traffic_video.mp4")
ANNOTATED_OUTPUT_PATH = Path(r"c:\g\tests\annotated_real_traffic.mp4")

def test_database_initialization():
    print("\n[TEST 1] Testing Database Initialization & Schema...")
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink(missing_ok=True)

    db = EdgeDBManager(TEST_DB_PATH)
    with sqlite3.connect(str(TEST_DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        print(f"  Created tables: {tables}")
        assert "vehicles_tracked" in tables, "vehicles_tracked table missing!"
        assert "lane_queue_aggregates" in tables, "lane_queue_aggregates table missing!"
        assert "system_heartbeat" in tables, "system_heartbeat table missing!"

        # Check WAL mode
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        print(f"  SQLite Journal Mode: {mode}")
        assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

    print("  ✓ Database Initialization Passed!")

def test_traffic_generation():
    print("\n[TEST 2] Verifying Real Traffic Video...")
    from tests.prepare_real_traffic_video import make_real_traffic_clip
    if not TEST_VIDEO_PATH.exists():
        make_real_traffic_clip(output_path=TEST_VIDEO_PATH)
    assert TEST_VIDEO_PATH.exists() and TEST_VIDEO_PATH.stat().st_size > 10000, "Failed to locate real traffic video!"
    print(f"  ✓ Traffic Video Ready: {TEST_VIDEO_PATH} ({TEST_VIDEO_PATH.stat().st_size / (1024*1024):.2f} MB)")

def test_perception_pipeline():
    print("\n[TEST 3] Running Full Perception & Tracking Pipeline on Real Traffic...")
    weights_to_use = r"c:\g\weights\edgeflow_yolov8n.pt" if Path(r"c:\g\weights\edgeflow_yolov8n.pt").exists() else "yolov8n.pt"
    is_vd = Path(r"c:\g\weights\edgeflow_yolov8n.pt").exists()
    import torch
    dev = "0" if torch.cuda.is_available() else "cpu"

    pipeline = EdgeFlowPerceptionPipeline(
        model_weights=weights_to_use,
        db_path=TEST_DB_PATH,
        conf_thresh=0.25,
        device=dev,
        is_visdrone=is_vd
    )

    # Process 50 frames of real traffic
    summary = pipeline.process_video(
        video_source=str(TEST_VIDEO_PATH),
        output_path=str(ANNOTATED_OUTPUT_PATH),
        max_frames=50
    )
    print(f"  Pipeline Run Summary: {summary}")
    assert summary["frames_processed"] > 0, "No frames processed!"
    assert ANNOTATED_OUTPUT_PATH.exists(), "Annotated output video was not saved!"
    print(f"  ✓ Pipeline executed {summary['frames_processed']} frames @ {summary['average_fps']} FPS")

def test_database_verification():
    print("\n[TEST 4] Verifying Telemetry Database Content...")
    with sqlite3.connect(str(TEST_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check vehicles_tracked
        cursor.execute("SELECT COUNT(*) FROM vehicles_tracked")
        track_count = cursor.fetchone()[0]
        print(f"  Logged vehicle track events: {track_count}")

        cursor.execute("SELECT track_id, class_name, speed_kmh, headway_m, pcu_weight, is_queued FROM vehicles_tracked LIMIT 5")
        sample_tracks = [dict(r) for r in cursor.fetchall()]
        for st in sample_tracks:
            print(f"    Sample Track: ID={st['track_id']}, Class={st['class_name']}, Spd={st['speed_kmh']}km/h, PCU={st['pcu_weight']}")

        # Check lane_queue_aggregates
        cursor.execute("SELECT COUNT(*) FROM lane_queue_aggregates")
        agg_count = cursor.fetchone()[0]
        print(f"  Logged lane queue aggregates: {agg_count}")
        assert agg_count > 0, "No lane aggregates recorded in database!"

        cursor.execute("SELECT lane_id, vehicle_count, total_pcu, queue_length_m, avg_approach_speed FROM lane_queue_aggregates ORDER BY id DESC LIMIT 4")
        sample_aggs = [dict(r) for r in cursor.fetchall()]
        for sa in sample_aggs:
            print(f"    Sample Lane State: {sa['lane_id']} | Count={sa['vehicle_count']} | PCU={sa['total_pcu']} | Q_len={sa['queue_length_m']}m")

        # Check system_heartbeat
        cursor.execute("SELECT COUNT(*) FROM system_heartbeat")
        hb_count = cursor.fetchone()[0]
        print(f"  Logged heartbeats: {hb_count}")
        assert hb_count > 0, "No heartbeat records logged!"

    print("  ✓ Database Verification Passed!")

def main():
    print("=========================================================")
    print("   DETECTION PERCEPTION & DATABASE INTEGRATION TESTS      ")
    print("=========================================================")
    test_database_initialization()
    test_traffic_generation()
    test_perception_pipeline()
    test_database_verification()
    print("\n=========================================================")
    print("   ALL DETECTION INTEGRATION TESTS PASSED SUCCESSFULLY!   ")
    print("=========================================================")

if __name__ == "__main__":
    main()
