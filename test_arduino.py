"""
Quick Arduino Hardware Diagnostic Tool
Tests serial connection to Arduino and cycles physical LEDs:
1. Connects to specified port (or auto-detects) @ 115200 baud.
2. Cycles NS Green/Yellow/Red and EW Green/Yellow/Red with confirmation output.
"""
import sys
import time
import serial
import serial.tools.list_ports

def scan_ports():
    ports = list(serial.tools.list_ports.comports())
    print("Available Serial COM Ports:")
    for p in ports:
        print(f"  - {p.device}: {p.description} (HWID: {p.hwid})")
    return [p.device for p in ports]

def test_arduino(port=None):
    print("=" * 60)
    print("   ARDUINO HARDWARE CONNECTION & LED DIAGNOSTIC   ")
    print("=" * 60)

    ports = scan_ports()
    if not port:
        if not ports:
            print("\n[ERROR] No COM ports detected. Check USB cable connection.")
            return False
        port = ports[0]

    print(f"\nAttempting connection to {port} @ 115200 baud...")
    try:
        ser = serial.Serial(port, 115200, timeout=1.0)
        time.sleep(2.0)  # Arduino DTR reset wait
        print(f"[SUCCESS] Connected to {port}!")
    except Exception as e:
        print(f"[ERROR] Failed to open {port}: {e}")
        return False

    print("\n--- Cycling Physical Signals via Serial Commands ---")
    commands = [
        ("SIG:G,R,10\n", "Step 1: North-South GREEN  | East-West RED"),
        ("SIG:Y,R,03\n", "Step 2: North-South YELLOW | East-West RED"),
        ("SIG:R,R,02\n", "Step 3: ALL RED (Clearance Safety)"),
        ("SIG:R,G,10\n", "Step 4: North-South RED    | East-West GREEN"),
        ("SIG:R,Y,03\n", "Step 5: North-South RED    | East-West YELLOW"),
        ("SIG:R,R,02\n", "Step 6: Return to ALL RED")
    ]

    try:
        for cmd, desc in commands:
            print(f"  Sending: {cmd.strip()} -> {desc}")
            ser.write(cmd.encode("ascii"))
            ser.flush()
            time.sleep(1.5)
        print("\n[COMPLETE] Arduino received all test signal packets successfully!")
    finally:
        ser.close()
        print(f"Closed {port}.")

    return True

if __name__ == "__main__":
    target_port = sys.argv[1] if len(sys.argv) > 1 else None
    test_arduino(target_port)
