/****************************************************************************
  FILENAME  ::   smooth_exact_sequence_accelstepper.ino
  AUTHORS   ::  CYRIL ANDRE DURANGO
                 SYKLER JASON BANZON
                 KINSHIN SORALLO

  PROGRAM DESCRIPTION ::
    AccelStepper-based exact-angle capture sequence with a hardened protocol:
      - Host sends: "on" \n, then <degrees>\n
      - Arduino echoes degrees as 4-byte LE, then WAITS for "go\n"
      - Only after "go", Arduino runs:
          startup spin (once per power), capture at 0°, then CW in increments:
            stop -> "capture\n" -> wait "picture captured" -> next
          finish at 360° (capture optional via CAPTURE_AT_360), send "exit\n"
          return 360° CCW
****************************************************************************/

#include <AccelStepper.h>

// ------------------- PINS -------------------
const int on_Nano = 12;
const int pythonOn = 11;
const int rst_pin = 5;   // A4988 RESET (active low)
const int slp_pin = 4;   // A4988 SLEEP (active low)
const int step_pin = 3;  // STEP
const int dir_pin  = 2;  // DIR

// ------------------- STATE -------------------
bool pythonConnected = false;
static int startup = 0;
long degrees_req = 0;

// ------------------- DIRECTION -------------------
// If your wiring spins the opposite way, flip CW_SIGN to -1.
const int CW_SIGN  = +1;
const int CCW_SIGN = -CW_SIGN;

// ------------------- MECHANICS -------------------
// Change to 3200 if using 1/16 microstepping.
const long STEPS_PER_REV = 200;

// ------------------- SPEED / ACCEL (tunable) -------------------
float MAX_SPEED_STEPS_PER_SEC = 50.0f;   // lower for heavier load
float ACCEL_STEPS_PER_SEC2    = 50.0f;   // lower for gentler ramps

// ------------------- SEQUENCE OPTIONS -------------------
#define CAPTURE_AT_360 0               // 0 = no capture at 360°, 1 = capture at 360° too
const unsigned FINAL_SETTLE_MS = 250;   // small pause if skipping 360 capture
const unsigned HOST_SWITCH_MS  = 150;   // let host switch to line mode

// ------------------- STEPPER -------------------
AccelStepper stepper(AccelStepper::DRIVER, step_pin, dir_pin);

// ------------------- HELPERS -------------------
void driverWake() {
  pinMode(rst_pin, OUTPUT);
  pinMode(slp_pin, OUTPUT);
  digitalWrite(slp_pin, HIGH); // wake
  digitalWrite(rst_pin, LOW);  delayMicroseconds(5);
  digitalWrite(rst_pin, HIGH); delayMicroseconds(5);
}

void stepperInit() {
  stepper.setMaxSpeed(MAX_SPEED_STEPS_PER_SEC);
  stepper.setAcceleration(ACCEL_STEPS_PER_SEC2);
  stepper.setCurrentPosition(0);
}

void moveStepsBlocking(long steps) {
  long target = stepper.currentPosition() + steps;
  stepper.moveTo(target);
  stepper.runToPosition(); // smooth accel/decel
}

void oneRevCW()  { moveStepsBlocking(CW_SIGN  * STEPS_PER_REV); }
void oneRevCCW() { moveStepsBlocking(CCW_SIGN * STEPS_PER_REV); }

// exact-angle planning with residual fix
int planDeltasForFullRev(long deg, long deltas[], int maxStops) {
  if (deg < 1) deg = 1;
  if (deg > 360) deg = 360;
  int N = (int)((360 + deg/2) / deg);
  if (N < 1) N = 1;
  if (N > maxStops) N = maxStops;

  long moved_prev = 0;
  for (int i = 1; i <= N; ++i) {
    long num = (long)i * deg * (long)STEPS_PER_REV;
    long tgt = (num + 180) / 360;
    if (tgt > STEPS_PER_REV) tgt = STEPS_PER_REV;
    deltas[i - 1] = tgt - moved_prev;
    moved_prev = tgt;
  }
  long sum = 0; for (int i = 0; i < N; ++i) sum += deltas[i];
  if (sum != STEPS_PER_REV) deltas[N - 1] += (STEPS_PER_REV - sum);
  return N;
}

void waitForAckLine() {
  while (Serial.available() == 0) {}
  (void)Serial.readStringUntil('\n'); // expect "picture captured"
}

void sendCaptureAndWait() {
  Serial.write("capture\n");
  waitForAckLine();
  Serial.write("Going to next capture interval\n");
}

// ------------------- SETUP / LOOP -------------------
void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50); // make parseInt reasonably snappy

  pinMode(on_Nano, OUTPUT);
  pinMode(pythonOn, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  pinMode(step_pin, OUTPUT);
  pinMode(dir_pin,  OUTPUT);
  pinMode(slp_pin,  OUTPUT);
  pinMode(rst_pin,  OUTPUT);

  digitalWrite(on_Nano, HIGH);
  digitalWrite(pythonOn, LOW);

  // driver idle
  digitalWrite(slp_pin, LOW);
  digitalWrite(rst_pin, LOW);

  // wait for host
  while (!Serial.available()) {}
}

void loop() {
  if (Serial.available() <= 0) return;

  // One-time startup spins (after we hear from host the first time)
  if (startup == 0) {
    driverWake();
    stepperInit();
    startup++;
  }

  String msg = Serial.readStringUntil('\n');

  if (msg == "HELLO") {
    pythonConnected = true;
    digitalWrite(pythonOn, HIGH);
    return;
  }
  if (!pythonConnected) return;

  if (msg.equalsIgnoreCase("on")) {
    // read angle next
    while (Serial.available() == 0) {}
    degrees_req = Serial.parseInt();

    // echo 4-byte LE confirmation (binary)
    Serial.write((byte*)&degrees_req, sizeof(degrees_req));
    Serial.flush();

    // give host time to read those 4 bytes and switch to line mode
    delay(HOST_SWITCH_MS);

    // *** WAIT for 'go' from the host before any 'capture' text ***
    while (true) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.length() == 0) continue;
      if (line.equalsIgnoreCase("go")) break; // now it's safe to send text lines
    }

    runCaptureSequence();
  }
}

void runCaptureSequence() {
  // Startup warmup happens once at boot. If you still want the CW/CCW warmup
  // each run, uncomment the next four lines:
   oneRevCW();  delay(500);
   oneRevCCW(); delay(500);

  // start at logical 0
  stepper.setCurrentPosition(0);

  // 0° capture
  sendCaptureAndWait();

  // plan CW increments
  const int MAX_STOPS = 360;
  long deltas[MAX_STOPS] = {0};
  int N = planDeltasForFullRev(degrees_req, deltas, MAX_STOPS);

  for (int i = 0; i < N; ++i) {
    moveStepsBlocking(CW_SIGN * deltas[i]);
    bool last = (i == N - 1);
    #if CAPTURE_AT_360
      sendCaptureAndWait();
    #else
      if (!last) sendCaptureAndWait();
      else       delay(FINAL_SETTLE_MS);
    #endif
  }

  Serial.write("exit\n");

  // return home
  delay(200);
  oneRevCCW();

  // optional: lower "connected" LED so UI knows we're idle
  pythonConnected = false;
  digitalWrite(pythonOn, LOW);
}
