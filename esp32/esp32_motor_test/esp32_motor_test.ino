// =============================================================
//  ESP32 Motor Test — No ROS, No Pi needed
//  Tests all 6 motors using 3x Cytron MDD10A drivers
//
//  Cycle: ALL FWD (2s) → STOP (1s) → ALL BWD (2s) → STOP (1s)
//         LEFT FWD + RIGHT BWD spin (1s) → STOP (1s)
//         LEFT BWD + RIGHT FWD spin (1s) → STOP (1s)
//
//  Watch each wheel and confirm all 6 spin.
//  Open Serial Monitor at 115200 to see status messages.
// =============================================================

#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ── Motor Pins (match your rover code exactly) ────────────────
#define FL_PWM 25
#define FL_DIR 26
#define FR_PWM 27
#define FR_DIR 14
#define ML_PWM 32
#define ML_DIR 33
#define MR_PWM 13
#define MR_DIR 12
#define RL_PWM 19
#define RL_DIR 18
#define RR_PWM 23
#define RR_DIR 4

// ── PWM Config ────────────────────────────────────────────────
#define PWM_FREQ    20000   // 20 kHz — inaudible, good for Cytron
#define PWM_RES     10      // 10-bit → 0–1023
#define PWM_MAX     1023
#define TEST_SPEED  600     // ~60% power — safe for bench test
                            // Raise to 800 if motors don't spin

// ── LEDC channel assignments ──────────────────────────────────
// ch 0=FL, 1=FR, 2=ML, 3=MR, 4=RL, 5=RR
void setupPWM() {
  ledcSetup(0, PWM_FREQ, PWM_RES); ledcAttachPin(FL_PWM, 0);
  ledcSetup(1, PWM_FREQ, PWM_RES); ledcAttachPin(FR_PWM, 1);
  ledcSetup(2, PWM_FREQ, PWM_RES); ledcAttachPin(ML_PWM, 2);
  ledcSetup(3, PWM_FREQ, PWM_RES); ledcAttachPin(MR_PWM, 3);
  ledcSetup(4, PWM_FREQ, PWM_RES); ledcAttachPin(RL_PWM, 4);
  ledcSetup(5, PWM_FREQ, PWM_RES); ledcAttachPin(RR_PWM, 5);
}

void setupDirPins() {
  pinMode(FL_DIR, OUTPUT);
  pinMode(FR_DIR, OUTPUT);
  pinMode(ML_DIR, OUTPUT);
  pinMode(MR_DIR, OUTPUT);
  pinMode(RL_DIR, OUTPUT);
  pinMode(RR_DIR, OUTPUT);
}

// ── Motor control ─────────────────────────────────────────────
// speed: positive = forward, negative = backward, 0 = stop
void setMotor(int ch, int dirPin, int speed) {
  speed = constrain(speed, -PWM_MAX, PWM_MAX);
  if (speed > 0) {
    digitalWrite(dirPin, LOW);
    ledcWrite(ch, speed);
  } else if (speed < 0) {
    digitalWrite(dirPin, HIGH);
    ledcWrite(ch, -speed);
  } else {
    ledcWrite(ch, 0);
  }
}

// Drive all 6 wheels: left side = leftSpeed, right side = rightSpeed
void driveAll(int leftSpeed, int rightSpeed) {
  setMotor(0, FL_DIR, leftSpeed);
  setMotor(2, ML_DIR, leftSpeed);
  setMotor(4, RL_DIR, leftSpeed);
  setMotor(1, FR_DIR, rightSpeed);
  setMotor(3, MR_DIR, rightSpeed);
  setMotor(5, RR_DIR, rightSpeed);
}

void stopAll() {
  driveAll(0, 0);
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  // Disable brownout detector — motor inrush can sag the 3.3V rail
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 Motor Test Starting ===");
  Serial.println("Watch all 6 wheels. Each test phase is labeled.");

  setupDirPins();
  setupPWM();
  stopAll();

  delay(2000);  // 2s pause before first move so you can step back
  Serial.println("Starting test sequence...");
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {

  // ── Phase 1: ALL FORWARD ──────────────────────────────────
  Serial.println("[1] ALL FORWARD — all 6 wheels should spin forward");
  driveAll(TEST_SPEED, TEST_SPEED);
  delay(2000);
  stopAll();
  Serial.println("    STOP");
  delay(1000);

  // ── Phase 2: ALL BACKWARD ─────────────────────────────────
  Serial.println("[2] ALL BACKWARD — all 6 wheels should spin backward");
  driveAll(-TEST_SPEED, -TEST_SPEED);
  delay(2000);
  stopAll();
  Serial.println("    STOP");
  delay(1000);

  // ── Phase 3: SPIN LEFT (left back, right fwd) ─────────────
  Serial.println("[3] SPIN LEFT — left wheels back, right wheels forward");
  driveAll(-TEST_SPEED, TEST_SPEED);
  delay(1500);
  stopAll();
  Serial.println("    STOP");
  delay(1000);

  // ── Phase 4: SPIN RIGHT (left fwd, right back) ────────────
  Serial.println("[4] SPIN RIGHT — left wheels forward, right wheels back");
  driveAll(TEST_SPEED, -TEST_SPEED);
  delay(1500);
  stopAll();
  Serial.println("    STOP");
  delay(1000);

  // ── Phase 5: LEFT SIDE ONLY ───────────────────────────────
  Serial.println("[5] LEFT ONLY — only FL, ML, RL should spin");
  setMotor(0, FL_DIR, TEST_SPEED);
  setMotor(2, ML_DIR, TEST_SPEED);
  setMotor(4, RL_DIR, TEST_SPEED);
  setMotor(1, FR_DIR, 0);
  setMotor(3, MR_DIR, 0);
  setMotor(5, RR_DIR, 0);
  delay(2000);
  stopAll();
  Serial.println("    STOP");
  delay(1000);

  // ── Phase 6: RIGHT SIDE ONLY ──────────────────────────────
  Serial.println("[6] RIGHT ONLY — only FR, MR, RR should spin");
  setMotor(0, FL_DIR, 0);
  setMotor(2, ML_DIR, 0);
  setMotor(4, RL_DIR, 0);
  setMotor(1, FR_DIR, TEST_SPEED);
  setMotor(3, MR_DIR, TEST_SPEED);
  setMotor(5, RR_DIR, TEST_SPEED);
  delay(2000);
  stopAll();
  Serial.println("    STOP");
  delay(2000);  // longer pause before repeating

  Serial.println("--- Repeating sequence ---\n");
}
