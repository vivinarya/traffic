/*
 * Traffic Light Vehicle Density Detector - Arduino Receiver & Hardware Simulator
 *
 * Works with OR without external LEDs:
 *   - Built-in LED (Pin 13 / LED_BUILTIN):
 *       * Solid ON  = North-South GREEN
 *       * Blinking  = North-South YELLOW
 *       * OFF       = North-South RED (East-West GREEN)
 *   - Optional External LEDs (if wired):
 *       * NS: Pins 2 (Red), 3 (Yellow), 4 (Green)
 *       * EW: Pins 5 (Red), 6 (Yellow), 7 (Green)
 *   - Serial Monitor Feedback @ 115200 baud:
 *       * Prints live visual traffic light status back to computer!
 */

const int PIN_BUILTIN   = 13;

const int PIN_NS_RED    = 2;
const int PIN_NS_YELLOW = 3;
const int PIN_NS_GREEN  = 4;

const int PIN_EW_RED    = 5;
const int PIN_EW_YELLOW = 6;
const int PIN_EW_GREEN  = 7;

String rxBuffer = "";
char currentNS = 'R';
char currentEW = 'R';
int currentTimer = 0;
unsigned long lastBlinkTime = 0;
bool blinkState = false;

void setup() {
  Serial.begin(115200);

  pinMode(PIN_BUILTIN, OUTPUT);
  pinMode(PIN_NS_RED, OUTPUT);
  pinMode(PIN_NS_YELLOW, OUTPUT);
  pinMode(PIN_NS_GREEN, OUTPUT);
  pinMode(PIN_EW_RED, OUTPUT);
  pinMode(PIN_EW_YELLOW, OUTPUT);
  pinMode(PIN_EW_GREEN, OUTPUT);

  // Startup confirmation
  setOutputs('R', 'R');
  Serial.println("ARDUINO_TRAFFIC_LIGHT_READY");
}

void loop() {
  // 1. Process incoming commands from Python detector
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(rxBuffer);
      rxBuffer = "";
    } else if (c != '\r') {
      rxBuffer += c;
    }
  }

  // 2. Built-in LED behavioral animation for standalone boards without external LEDs
  if (currentNS == 'G') {
    digitalWrite(PIN_BUILTIN, HIGH); // Solid ON when NS has Green
  } else if (currentNS == 'Y' || currentEW == 'Y') {
    // Fast blinking during Yellow clearance
    if (millis() - lastBlinkTime >= 150) {
      lastBlinkTime = millis();
      blinkState = !blinkState;
      digitalWrite(PIN_BUILTIN, blinkState ? HIGH : LOW);
    }
  } else {
    // NS is Red (EW is Green) -> slow heartbeat pulse
    if (millis() - lastBlinkTime >= 600) {
      lastBlinkTime = millis();
      blinkState = !blinkState;
      digitalWrite(PIN_BUILTIN, blinkState ? HIGH : LOW);
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  // Expected format: SIG:<NS>,<EW>,<TIMER> e.g. SIG:G,R,18
  if (cmd.startsWith("SIG:")) {
    String payload = cmd.substring(4);
    int firstComma = payload.indexOf(',');
    int secondComma = payload.indexOf(',', firstComma + 1);

    if (firstComma != -1) {
      currentNS = payload.charAt(0);
      currentEW = payload.charAt(firstComma + 1);
      if (secondComma != -1) {
        currentTimer = payload.substring(secondComma + 1).toInt();
      }

      setOutputs(currentNS, currentEW);

      // Send feedback back to Python
      Serial.print("ACK: NS=");
      Serial.print(currentNS == 'G' ? "GREEN" : (currentNS == 'Y' ? "YELLOW" : "RED"));
      Serial.print(", EW=");
      Serial.print(currentEW == 'G' ? "GREEN" : (currentEW == 'Y' ? "YELLOW" : "RED"));
      Serial.print(" [");
      Serial.print(currentTimer);
      Serial.println("s]");
    }
  }
}

void setOutputs(char nsState, char ewState) {
  // North-South LEDs
  digitalWrite(PIN_NS_RED,    (nsState == 'R') ? HIGH : LOW);
  digitalWrite(PIN_NS_YELLOW, (nsState == 'Y') ? HIGH : LOW);
  digitalWrite(PIN_NS_GREEN,  (nsState == 'G') ? HIGH : LOW);

  // East-West LEDs
  digitalWrite(PIN_EW_RED,    (ewState == 'R') ? HIGH : LOW);
  digitalWrite(PIN_EW_YELLOW, (ewState == 'Y') ? HIGH : LOW);
  digitalWrite(PIN_EW_GREEN,  (ewState == 'G') ? HIGH : LOW);
}
