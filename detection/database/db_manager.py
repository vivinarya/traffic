"""
Detection Telemetry Database Manager
High-performance SQLite engine with WAL mode designed for Edge SBC deployment.
Handles batched writes from tracking and rapid state retrieval for optimization.
"""
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_DEFAULT_PATH = Path(r"c:\g\edgeflow_telemetry.db")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

class EdgeDBManager:
    def __init__(self, db_path: Path = DB_DEFAULT_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for concurrency and fast writes on edge flash media
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        return conn

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            if SCHEMA_PATH.exists():
                with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                    conn.executescript(f.read())
            conn.commit()

    def log_tracks_batch(self, tracks: List[Dict[str, Any]]):
        """
        Inserts a batch of vehicle tracking observations.
        Each track dict: {timestamp, frame_id, track_id, class_id, class_name, confidence,
                          bbox_x1, bbox_y1, bbox_x2, bbox_y2, lane_id, speed_kmh, headway_m, pcu_weight, is_queued}
        """
        if not tracks:
            return

        query = """
        INSERT INTO vehicles_tracked (
            timestamp, frame_id, track_id, class_id, class_name, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, lane_id, speed_kmh, headway_m, pcu_weight, is_queued
        ) VALUES (
            :timestamp, :frame_id, :track_id, :class_id, :class_name, :confidence,
            :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2, :lane_id, :speed_kmh, :headway_m, :pcu_weight, :is_queued
        )
        """
        with self._get_connection() as conn:
            conn.executemany(query, tracks)
            conn.commit()

    def log_lane_aggregates(self, aggregates: List[Dict[str, Any]]):
        """
        Inserts aggregate lane metrics for the optimizer.
        Each item: {timestamp, lane_id, vehicle_count, total_pcu, queue_length_m, avg_approach_speed, lead_vehicle_headway}
        """
        if not aggregates:
            return

        query = """
        INSERT INTO lane_queue_aggregates (
            timestamp, lane_id, vehicle_count, total_pcu, queue_length_m, avg_approach_speed, lead_vehicle_headway
        ) VALUES (
            :timestamp, :lane_id, :vehicle_count, :total_pcu, :queue_length_m, :avg_approach_speed, :lead_vehicle_headway
        )
        """
        with self._get_connection() as conn:
            conn.executemany(query, aggregates)
            conn.commit()

    def log_heartbeat(self, fps: float, det_ms: float, track_ms: float, total_ms: float, cpu_pct: float = 0.0, mcu_status: str = "OK"):
        query = """
        INSERT INTO system_heartbeat (
            timestamp, inference_fps, detection_latency_ms, tracking_latency_ms,
            total_pipeline_latency_ms, cpu_percent, mcu_link_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(query, (time.time(), fps, det_ms, track_ms, total_ms, cpu_pct, mcu_status))
            conn.commit()

    def get_latest_lane_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetches the most recent state of all approach lanes for the optimization solver.
        """
        query = """
        SELECT lane_id, vehicle_count, total_pcu, queue_length_m, avg_approach_speed, lead_vehicle_headway, MAX(timestamp) as last_seen
        FROM lane_queue_aggregates
        GROUP BY lane_id
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            return {row["lane_id"]: dict(row) for row in rows}

    def prune_old_telemetry(self, retention_hours: int = 48):
        """Prunes raw frame records older than retention_hours to prevent disk growth."""
        cutoff = time.time() - (retention_hours * 3600)
        with self._get_connection() as conn:
            conn.execute("DELETE FROM vehicles_tracked WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM system_heartbeat WHERE timestamp < ?", (cutoff,))
            conn.commit()
            conn.execute("VACUUM;")
