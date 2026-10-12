-- EdgeFlow Database Schema
-- Embedded SQLite optimized for Edge SBC (Jetson Nano / Raspberry Pi 5)
-- Pragma settings: WAL mode, synchronous=NORMAL, cache_size=-64000

-- 1. Detailed Vehicle Tracking Events
CREATE TABLE IF NOT EXISTS vehicles_tracked (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,              -- Epoch timestamp (float seconds)
    frame_id INTEGER NOT NULL,            -- Ingest video frame counter
    track_id INTEGER NOT NULL,            -- Persistent ByteTrack ID
    class_id INTEGER NOT NULL,            -- Detected class index (0-9)
    class_name TEXT NOT NULL,             -- car, bus, truck, motor, etc.
    confidence REAL NOT NULL,             -- Detector confidence score (0.0 - 1.0)
    bbox_x1 REAL NOT NULL,                -- Normalized bounding box [x1, y1, x2, y2]
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    lane_id TEXT DEFAULT 'Approach_Default', -- Assigned approach corridor
    speed_kmh REAL DEFAULT 0.0,           -- Estimated instantaneous velocity
    headway_m REAL DEFAULT 0.0,           -- Spatial gap to front vehicle (meters)
    pcu_weight REAL NOT NULL DEFAULT 1.0, -- Passenger Car Unit weight (e.g. 1.0, 0.5, 3.0)
    is_queued INTEGER DEFAULT 0           -- 1 if queued behind stopline, 0 if moving
);

CREATE INDEX IF NOT EXISTS idx_vehicles_track_time ON vehicles_tracked(timestamp, track_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_lane ON vehicles_tracked(lane_id, timestamp);

-- 2. Rolling Horizon Lane Queue & Approach Aggregates
-- Feed directly into the constrained optimization engine (MILP / Delay Minimization)
CREATE TABLE IF NOT EXISTS lane_queue_aggregates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,              -- Snapshot time
    lane_id TEXT NOT NULL,                -- Approach lane identifier
    vehicle_count INTEGER NOT NULL,       -- Raw count of vehicles in corridor
    total_pcu REAL NOT NULL,              -- Weighted Passenger Car Units sum
    queue_length_m REAL NOT NULL,         -- Length of physical queue in meters
    avg_approach_speed REAL NOT NULL,     -- Mean velocity of approaching platoon (km/h)
    lead_vehicle_headway REAL DEFAULT 0.0 -- Headway of leading vehicle to stopline
);

CREATE INDEX IF NOT EXISTS idx_aggregates_lane_time ON lane_queue_aggregates(lane_id, timestamp);

-- 3. Edge System Heartbeat & Watchdog Telemetry
-- Monitored by failsafe MCU (ESP32/STM32) to ensure edge inference is active
CREATE TABLE IF NOT EXISTS system_heartbeat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    inference_fps REAL NOT NULL,
    detection_latency_ms REAL NOT NULL,
    tracking_latency_ms REAL NOT NULL,
    total_pipeline_latency_ms REAL NOT NULL,
    cpu_percent REAL DEFAULT 0.0,
    mcu_link_status TEXT DEFAULT 'OK'
);

CREATE INDEX IF NOT EXISTS idx_heartbeat_time ON system_heartbeat(timestamp);
