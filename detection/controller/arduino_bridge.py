"""
Traffic Light Vehicle Density Detector - Arduino Serial Hardware Bridge
Manages USB serial communication with a connected Arduino microcontroller.
Sends real-time signal commands:
"SIG:<NS_STATE>,<EW_STATE>,<COUNTDOWN>\n" (e.g. "SIG:G,R,18\n")
Operates non-blocking in a background thread to prevent inference stutter.
"""
import time
import queue
import threading
from typing import Optional, List, Dict
import serial
import serial.tools.list_ports

class ArduinoTrafficLightBridge:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.status_text = "SIMULATION MODE"

        self.tx_queue: queue.Queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.last_sent_state = ""

        self._initialize_connection()
        if self.is_connected:
            self.worker_thread = threading.Thread(target=self._tx_worker, daemon=True)
            self.worker_thread.start()

    def _initialize_connection(self):
        """Discovers and opens serial port."""
        if not self.port:
            self.port = self._auto_detect_arduino_port()

        if self.port:
            try:
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
                time.sleep(1.5)  # Allow Arduino DTR reset settle time
                self.is_connected = True
                self.status_text = f"CONNECTED ({self.port})"
                print(f"[Arduino Bridge] Successfully connected to Arduino on {self.port} @ {self.baudrate} baud.")
            except Exception as e:
                self.is_connected = False
                self.status_text = "SIMULATION MODE"
                print(f"[Arduino Bridge] Port {self.port} unavailable ({e}). Operating in Software Simulation Mode.")
        else:
            self.is_connected = False
            self.status_text = "SIMULATION MODE"
            print("[Arduino Bridge] No physical Arduino detected. Operating in Software Simulation Mode.")

    def _auto_detect_arduino_port(self) -> Optional[str]:
        """Scans for Arduino / CH340 / CP2102 / FTDI USB-Serial devices."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            mfg = (p.manufacturer or "").lower()
            if any(k in desc or k in mfg for k in ["arduino", "ch340", "cp210", "ftdi", "usb serial"]):
                return p.device
        # Return first available COM port if only one exists
        if len(ports) == 1:
            return ports[0].device
        return None

    def send_signal_state(self, ns_light: str, ew_light: str, countdown_sec: int):
        """
        Sends compact signal packet to Arduino.
        ns_light: 'GREEN', 'YELLOW', 'RED' -> 'G', 'Y', 'R'
        ew_light: 'GREEN', 'YELLOW', 'RED' -> 'G', 'Y', 'R'
        countdown_sec: int
        """
        if not self.is_connected or not self.serial_conn:
            return

        c_ns = ns_light[0].upper()
        c_ew = ew_light[0].upper()
        payload = f"SIG:{c_ns},{c_ew},{countdown_sec}\n"

        # Rate-limit duplicate packets to save MCU cycle time
        if payload != self.last_sent_state:
            self.last_sent_state = payload
            try:
                if not self.tx_queue.full():
                    self.tx_queue.put_nowait(payload)
                    print(f"  [Arduino Serial TX] -> {payload.strip()} (NS: {ns_light}, EW: {ew_light}, Timer: {countdown_sec}s)")
            except queue.Full:
                pass

    def _tx_worker(self):
        """Background thread writing bytes and reading ACK responses from serial."""
        while not self.stop_event.is_set():
            try:
                # 1. Read any incoming bytes from Arduino
                if self.serial_conn and self.serial_conn.is_open and self.serial_conn.in_waiting > 0:
                    try:
                        rx_line = self.serial_conn.readline().decode("ascii", errors="ignore").strip()
                        if rx_line:
                            print(f"  [Arduino Hardware RX] <- {rx_line}")
                    except Exception:
                        pass

                # 2. Transmit queued packets
                try:
                    payload = self.tx_queue.get(timeout=0.1)
                    if self.serial_conn and self.serial_conn.is_open:
                        self.serial_conn.write(payload.encode("ascii"))
                        self.serial_conn.flush()
                except queue.Empty:
                    pass
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Arduino Bridge] Serial transmission error: {e}")
                self.is_connected = False
                self.status_text = "DISCONNECTED"
                break

    def close(self):
        self.stop_event.set()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.is_connected = False
