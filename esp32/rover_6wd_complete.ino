/*
 * rover_6wd_complete.ino
 *
 * Micro-ROS firmware for 6-wheel differential drive rover.
 *
 * KEY FEATURES
 *   1. Full agent reconnection state machine — the ESP32 recovers when the Pi
 *      reboots, the agent crashes, or the USB cable glitches.
 *   2. Wheel encoders (NEW): one quadrature encoder per side (mid wheel).
 *        • Signed odometry — /rover/rpm now carries REAL, direction-aware RPM
 *          consumed by rpm_to_odom.py → /odom + TF (enables SLAM/EKF, goal
 *          distance, drift-free-ish dead reckoning).
 *        • Closed-loop velocity PID — commanded /cmd_vel wheel speeds are now
 *          ENFORCED by feedback, not guessed open-loop. Fixes battery-sag
 *          slowdown, terrain load, and asymmetric left/right turn response.
 *
 * ROS2 interface (positions 0,1 UNCHANGED — rpm_to_odom.py depends on this
 * contract; positions 2,3 are NEW additive fields, see IMU note below):
 *   Subscribes : /cmd_vel   (geometry_msgs/Twist)
 *   Publishes  : /rover/rpm  (std_msgs/Float32MultiArray
 *                             [left_rpm, right_rpm, imu_heading_deg, imu_calib],
 *                             signed RPM: +ve = wheel drives robot FORWARD;
 *                             imu_heading_deg: BNO055 fused absolute heading
 *                             in degrees [0,360), or NaN if the IMU didn't
 *                             ack at startup; imu_calib: human-readable
 *                             SYS*1000 + GYR*100 + ACC*10 + MAG, each digit
 *                             0(uncalibrated)-3(fully calibrated), e.g. 3320
 *                             = SYS 3, GYR 3, ACC 2, MAG 0 -- existing
 *                             consumers that only read data[0]/data[1]
 *                             (len(data) >= 2 checks) are unaffected by the
 *                             array growing to 4.
 *
 * IMU (NEW): BNO055 on I2C (GPIO21 SDA / GPIO22 SCL, ESP32 defaults -- free
 * on this board, no conflict with the motor/encoder pins above). Wheel-
 * differential heading (theta) drifts sharply past ~135-165deg of rotation
 * on this skid-steer chassis (measured, see odometry_log/
 * odom_accuracy_results.csv) because turning scrubs the wheels against the
 * ground -- the encoders can't see that slip. BNO055 does its own onboard
 * sensor fusion (accel+gyro+mag) and reports an absolute heading that
 * doesn't depend on wheel odometry at all, so it's immune to that specific
 * failure mode. Raw I2C register access (no Adafruit_BNO055/Adafruit_Sensor
 * library dependency) -- see bno055_init()/bno055_read_heading_deg() below.
 * Calibration offsets persist across power cycles (NVS via Preferences,
 * still no new external library): the first time this boot's calibration
 * reaches SYS/GYR/ACC/MAG all == 3, bno055_save_calibration() stores the
 * offset block to flash; every future boot's bno055_init() restores it via
 * bno055_load_calibration(), skipping the physical wave-around calibration
 * dance. Getting CALIBRATED (Pi-side theta_src == "imu") does NOT mean the
 * heading is actually ACCURATE, though: --imu-min-mag-calib on the Pi side
 * (home_gui.py et al) gates on the MAG sub-score alone, and level 1 there
 * (the default) is a low bar -- real magnetic disturbance near the sensor
 * (motor current, nearby metal/wiring) can still swing the reported heading
 * tens of degrees at MAG==1. Level 3 is Bosch's own bar for a trustworthy
 * absolute heading; if MAG can't reach 3 in a given room, that's a real
 * environment/mounting issue this persistence feature doesn't paper over.
 *
 * SIGN CONVENTION
 *   The firmware owns the encoder sign so that a forward command yields
 *   POSITIVE measured RPM on both sides. Therefore rpm_to_odom.py should keep
 *   left_sign = right_sign = +1. If a side reads backwards on the bench, flip
 *   ENC_L_SIGN / ENC_R_SIGN below (compile-time) — do NOT also flip it in the
 *   Python node, or the two negations cancel.
 */

#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <Wire.h>
#include <Preferences.h>  // ESP32 core NVS wrapper (bundled with the core, same tier as
                          // Wire.h -- not a new external library, see BNO055 calibration
                          // persistence note below)

// ─────────────────────────────────────────────────────────────────────────────
// MOTOR PINS (6-wheel rover: 3 motors per side)
// ─────────────────────────────────────────────────────────────────────────────
#define FL_PWM  25
#define FL_DIR  26
#define FR_PWM  27
#define FR_DIR  14

#define ML_PWM  32
#define ML_DIR  33
#define MR_PWM  13
#define MR_DIR  12

#define RL_PWM  19
#define RL_DIR  18
#define RR_PWM  23
#define RR_DIR  4

// ─────────────────────────────────────────────────────────────────────────────
// ENCODER PINS  (one quadrature encoder per side, on the mid wheel)
//   ENC_x_A → interrupt (RISING);  ENC_x_B → direction (quadrature phase)
//   GPIO 34/35/36/39 are INPUT-ONLY on the ESP32 and have NO internal pull-ups.
//   The encoder outputs must be push-pull (Hall/optical with driver) OR you
//   must add external pull-ups. These pins are interrupt-capable.
// ─────────────────────────────────────────────────────────────────────────────
#define ENC_L_A  36
#define ENC_L_B  39
#define ENC_R_A  34
#define ENC_R_B  35

// ─────────────────────────────────────────────────────────────────────────────
// ROVER PARAMETERS
// ─────────────────────────────────────────────────────────────────────────────
#define WHEEL_RADIUS_M      0.056f   // m  — must match rpm_to_odom.py wheel_radius
#define TRACK_WIDTH_M       0.345f   // m  — must match rpm_to_odom.py wheel_separation
#define MAX_SPEED_MS        0.7f     // m/s that maps to full PWM (feedforward). Raised
                                     // 0.5->0.7 to slow the rover: a given command now
                                     // maps to a lower PWM (ratio = vel/MAX_SPEED_MS).
#define CMD_VEL_TIMEOUT_MS  500      // zero motors after 500 ms of /cmd_vel silence
#define RPM_PUBLISH_MS      100      // 10 Hz odometry publish (matches old contract)

// Encoder counts per WHEEL revolution as seen by this firmware.
//   Current wiring counts 1 edge (channel A RISING) per encoder pulse → 1x.
//   ENCODER_CPR is the number of A-rising edges per full WHEEL turn.
//   ┌ IMPORTANT: if the encoder sits on the MOTOR shaft (before a gearbox),
//   │ set this to (pulses_per_motor_rev × gear_ratio). Wrong value = wrong
//   │ odometry scale AND wrong closed-loop speed. Tune on the bench: push the
//   │ rover exactly 1 wheel turn and read the count.
// 2026-07-09 bench calibration: powered spin, right mid wheel, 37572 counts /
// 11.5 wheel revs = 3267 counts/rev (Johnson 100RPM quad encoder on motor shaft).
#define ENCODER_CPR         3267.0f

// Per-side encoder polarity (+1 or -1). Flip if a side reports negative RPM
// while physically rolling forward. See SIGN CONVENTION header note.
// 2026-07-09 bench check: forward cmd read NEGATIVE on both sides → both -1.
#define ENC_L_SIGN          (-1.0f)
#define ENC_R_SIGN          (-1.0f)

#define AGENT_PING_TIMEOUT_MS  500
#define AGENT_PING_ATTEMPTS     20

// PWM (LEDC) configuration
#define PWM_FREQ   20000
#define PWM_RES    10
#define PWM_MAX    1023
#define MIN_PWM    380      // motors stall below this; feedforward floor. Lowered
                            // 450->380 to reduce the hard minimum speed. If a wheel
                            // stalls/buzzes at low command, raise back toward ~410.

// ─────────────────────────────────────────────────────────────────────────────
// CLOSED-LOOP VELOCITY CONTROL
//   output_pwm = feedforward(target) + PI_correction(target - measured)
//   Feedforward does the heavy lifting; PI trims. Correction is BOUNDED so a
//   miswired/failed encoder can never cause full-scale runaway.
// ─────────────────────────────────────────────────────────────────────────────
#define CLOSED_LOOP        1        // master switch: 0 = pure open-loop feedforward
#define CONTROL_HZ         50       // control loop rate
#define CONTROL_DT_MS      (1000 / CONTROL_HZ)

#define VEL_DEADBAND_MS    0.03f    // |target| below this → hard stop that side
#define KP_PWM_PER_MS      600.0f   // proportional: PWM per (m/s) of error
#define KI_PWM_PER_MS      250.0f   // integral:     PWM per (m/s·s) of error
#define CORR_CLAMP         350.0f   // max |PI correction| (PWM units) — runaway cap
#define I_CLAMP            (CORR_CLAMP / KI_PWM_PER_MS)  // anti-windup on integrator
#define VEL_EMA_ALPHA      0.4f     // measured-velocity smoothing (0..1, higher=faster)

// Runaway / fault guard: if a side is commanded hard for GUARD_MS but the
// encoder reads dead (~0) or the WRONG sign, that side latches to open-loop
// (feedforward only) so PID feedback can never drive it to full scale.
#define GUARD_MS           700
#define GUARD_CMD_MS       0.12f    // "commanded hard" threshold (m/s)
#define GUARD_MIN_MEAS_MS  0.02f    // below this while commanded = dead encoder

// ─────────────────────────────────────────────────────────────────────────────
// RECONNECTION STATE MACHINE
// ─────────────────────────────────────────────────────────────────────────────
enum AgentState { WAITING_AGENT, AGENT_CONNECTED, AGENT_DISCONNECTED };
AgentState agent_state = WAITING_AGENT;

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL STATE
// ─────────────────────────────────────────────────────────────────────────────
// Monotonic signed encoder counters (raw, direction from channel B).
// TWO independent consumers read these (control loop + RPM publisher), each
// keeping its own snapshot, so neither resets the shared counter.
volatile long enc_left_count  = 0;
volatile long enc_right_count = 0;

unsigned long last_publish_time = 0;
unsigned long last_cmd_vel_time = 0;
float cmd_linear_x  = 0.0f;
float cmd_angular_z = 0.0f;

// ─────────────────────────────────────────────────────────────────────────────
// ROS2 OBJECTS
// ─────────────────────────────────────────────────────────────────────────────
rcl_subscription_t  cmd_vel_sub;
rcl_publisher_t     rpm_pub;
rclc_support_t      support;
rcl_allocator_t     allocator;
rcl_node_t          node;
rclc_executor_t     executor;

geometry_msgs__msg__Twist         cmd_vel_msg;
std_msgs__msg__Float32MultiArray  rpm_msg;
float rpm_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // [left_rpm, right_rpm, imu_heading_deg, imu_calib]

// ─────────────────────────────────────────────────────────────────────────────
// PER-SIDE CONTROL STATE
// ─────────────────────────────────────────────────────────────────────────────
struct SideCtrl {
  long   last_count;      // encoder snapshot for the control loop
  float  meas_vel_ms;     // EMA-filtered measured wheel speed (m/s)
  float  integ;           // PI integrator
  bool   openloop_latch;  // true → feedforward only (encoder faulted)
  unsigned long fault_ms; // how long the guard condition has held
};
SideCtrl L = {0, 0.0f, 0.0f, false, 0};
SideCtrl R = {0, 0.0f, 0.0f, false, 0};

// meters travelled by the wheel rim per encoder count
static const float METERS_PER_COUNT = (2.0f * PI * WHEEL_RADIUS_M) / ENCODER_CPR;

// snapshot for the 10 Hz RPM publisher
long   pub_last_left  = 0;
long   pub_last_right = 0;

// ─────────────────────────────────────────────────────────────────────────────
// BNO055 IMU (I2C, absolute-heading fusion) — see the file header note above.
// Raw register access, no external library: keeps this sketch's dependency
// footprint at just Wire.h (already part of the ESP32 core), so it can't
// introduce a NEW library-version mismatch on top of the existing
// micro_ros_arduino/esp32-core pinning (see esp32-flashing-procedure memo).
// ─────────────────────────────────────────────────────────────────────────────
#define BNO055_I2C_ADDR             0x28   // default (ADR pin low/unconnected)
#define BNO055_SDA_PIN              21
#define BNO055_SCL_PIN              22

#define BNO055_REG_CHIP_ID          0x00
#define BNO055_CHIP_ID_VALUE        0xA0
#define BNO055_REG_PAGE_ID          0x07
#define BNO055_REG_EUL_HEADING_LSB  0x1A
#define BNO055_REG_CALIB_STAT       0x35
#define BNO055_REG_OPR_MODE         0x3D
#define BNO055_REG_PWR_MODE         0x3E

// Sensor offset/radius block (accel offset x/y/z, mag offset x/y/z, gyro
// offset x/y/z, accel radius, mag radius -- BNO055 datasheet table 3-42),
// 22 contiguous bytes. Only readable/writable in CONFIG mode. This is the
// standard save/restore-calibration mechanism (Bosch datasheet section
// 3.6.4): read it out once fully calibrated, persist it, and write it back
// on every future boot instead of re-doing the physical wave-around
// calibration dance each time -- see bno055_save_calibration()/
// bno055_load_calibration() below.
#define BNO055_REG_CALIB_START      0x55
#define BNO055_CALIB_LEN            22

// Bosch's NDOF fusion algorithm reports the MAG sub-score conservatively:
// even with a known-good offset profile just restored from flash, it won't
// report mag calibrated (score 3) until it has seen live, consistent
// magnetometer readings post-boot -- unlike accel/gyro, whose calibration
// is sensor-intrinsic (bias/temperature) and validates almost instantly,
// mag calibration compensates for LOCAL magnetic interference, which the
// chip has no way to know hasn't changed since the offsets were saved. Set
// this to 1 to bridge JUST that initial post-boot gap: report mag as
// trusted (score 3) immediately after a saved profile loads, but ONLY
// until the live status genuinely reaches 3 on its own for the first time
// this boot (bno_mag_live_confirmed) -- from that point on the override is
// inert and the live reading is fully authoritative again, so a real
// disturbance encountered later during actual use (motor EMI while
// driving, reproduced 2026-08-06: heading drifted 24 deg in ~300ms while
// stationary right after a drive burst) still correctly falls back to
// wheel-diff instead of being permanently masked. Safe ONLY if the rover
// stays in the same magnetic environment (same room, same nearby
// metal/motors/wiring) it was calibrated in -- if it's moved somewhere
// very different, the bridged window right after boot could mask a bad
// heading, same risk as before, just narrowed to that one window instead
// of the whole session (see nav_pipeline/odometry_logger.py's
// imu_min_mag_calib gate, Pi-side).
#define TRUST_LOADED_MAG_CALIB      1

#define BNO055_MODE_CONFIG          0x00
#define BNO055_MODE_NDOF            0x0C
#define BNO055_PWR_NORMAL           0x00

bool imu_present = false;
Preferences bno_prefs;
bool bno_calib_saved_this_boot = false;  // avoid hammering flash every publish tick
                                          // once full calibration is reached -- see loop()
bool bno_calib_loaded_this_boot = false; // a saved offset block was found + applied at boot
                                          // -- see bno055_load_calibration()/TRUST_LOADED_MAG_CALIB
bool bno_mag_live_confirmed = false;     // raw MAG sub-score has genuinely reached 3 at least
                                          // once THIS boot -- once true, TRUST_LOADED_MAG_CALIB
                                          // stops bridging and the live status is fully
                                          // authoritative again, so a real disturbance after that
                                          // point (e.g. motor EMI while driving) still correctly
                                          // falls back to wheel-diff instead of being masked.

bool bno055_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(BNO055_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool bno055_read(uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(BNO055_I2C_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;  // repeated START, keep bus held
  if (Wire.requestFrom((int)BNO055_I2C_ADDR, (int)len) != (int)len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

// Restore a previously-saved calibration offset block (see
// bno055_save_calibration() below) -- MUST be called while already in
// CONFIG mode (offset registers aren't accessible in NDOF). No-op if
// nothing was ever saved: a fresh board, or one that's never yet completed
// a full physical calibration, just keeps the sensor's power-on defaults
// and calibrates from scratch as before. Sets bno_calib_loaded_this_boot
// on success -- see TRUST_LOADED_MAG_CALIB below for what that unlocks.
void bno055_load_calibration() {
  bno_prefs.begin("bno055", true);  // read-only
  if (bno_prefs.getBytesLength("offsets") == BNO055_CALIB_LEN) {
    uint8_t buf[BNO055_CALIB_LEN];
    bno_prefs.getBytes("offsets", buf, BNO055_CALIB_LEN);
    for (uint8_t i = 0; i < BNO055_CALIB_LEN; i++) {
      bno055_write8(BNO055_REG_CALIB_START + i, buf[i]);
    }
    bno_calib_loaded_this_boot = true;
  }
  bno_prefs.end();
}

// Persist the current calibration offset block to flash (NVS via
// Preferences) so a future boot can skip the physical wave-around
// calibration dance entirely -- see bno055_load_calibration() and loop()'s
// RPM-publish block, which calls this once per boot the first time
// bno055_read_calib_status() reports full calibration (0xFF). Briefly
// switches to CONFIG mode to read the offset registers (only accessible
// there, datasheet section 3.6.4) -- a one-time ~50ms fusion-output pause
// the very first time full calibration is reached each boot, never again
// after that (guarded by bno_calib_saved_this_boot), so it doesn't
// meaningfully compete with the control loop's cadence.
void bno055_save_calibration() {
  bno055_write8(BNO055_REG_PAGE_ID, 0x00);
  bno055_write8(BNO055_REG_OPR_MODE, BNO055_MODE_CONFIG);
  delay(25);

  uint8_t buf[BNO055_CALIB_LEN];
  if (bno055_read(BNO055_REG_CALIB_START, buf, BNO055_CALIB_LEN)) {
    bno_prefs.begin("bno055", false);  // read/write
    bno_prefs.putBytes("offsets", buf, BNO055_CALIB_LEN);
    bno_prefs.end();
  }

  bno055_write8(BNO055_REG_OPR_MODE, BNO055_MODE_NDOF);
  delay(25);
}

// Returns false if the BNO055 never acks/identifies correctly -- this IS the
// wiring check: CHIP_ID only reads back 0xA0 if VIN/GND/SDA/SCL are all
// actually connected and the sensor is powered.
bool bno055_init() {
  Wire.begin(BNO055_SDA_PIN, BNO055_SCL_PIN);
  Wire.setClock(100000);  // 100kHz standard-mode I2C -- safe default, no need
                           // to risk 400kHz fast-mode on an unverified bus

  uint8_t chip_id = 0;
  if (!bno055_read(BNO055_REG_CHIP_ID, &chip_id, 1) || chip_id != BNO055_CHIP_ID_VALUE) {
    return false;
  }

  bno055_write8(BNO055_REG_PAGE_ID, 0x00);
  bno055_write8(BNO055_REG_OPR_MODE, BNO055_MODE_CONFIG);
  delay(25);                                    // mode-switch settle (datasheet: 19ms to CONFIG)
  bno055_write8(BNO055_REG_PWR_MODE, BNO055_PWR_NORMAL);
  bno055_write8(BNO055_REG_PAGE_ID, 0x00);
  bno055_load_calibration();  // restore a saved offset block, if any (must run in CONFIG mode)
  bno055_write8(BNO055_REG_OPR_MODE, BNO055_MODE_NDOF);  // full 9-DOF fusion, absolute heading
  delay(25);                                    // mode-switch settle (datasheet: 7ms, +margin)
  return true;
}

// Fused absolute heading, degrees [0,360). NaN if the IMU never initialized
// or a read fails (transient I2C glitch) -- callers must NaN-check, never
// assume this is always valid.
float bno055_read_heading_deg() {
  if (!imu_present) return NAN;
  uint8_t buf[2];
  if (!bno055_read(BNO055_REG_EUL_HEADING_LSB, buf, 2)) return NAN;
  int16_t raw = (int16_t)((uint16_t)buf[1] << 8 | buf[0]);
  return raw / 16.0f;  // 1 LSB = 1/16 degree (BNO055 datasheet Table 3-22)
}

// SYS/GYRO/ACCEL/MAG calibration, each 0 (uncalibrated) - 3 (fully
// calibrated). Not currently published -- available for future use (e.g.
// gating theta-snap corrections on the Pi side until SYS reaches 3).
uint8_t bno055_read_calib_status() {
  uint8_t v = 0;
  if (!imu_present || !bno055_read(BNO055_REG_CALIB_STAT, &v, 1)) return 0;
  return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// ENCODER ISRs  (1x quadrature: count on channel-A rising, phase from B)
// ─────────────────────────────────────────────────────────────────────────────
void IRAM_ATTR enc_left_isr() {
  if (digitalRead(ENC_L_B) == HIGH) enc_left_count++;
  else                              enc_left_count--;
}
void IRAM_ATTR enc_right_isr() {
  if (digitalRead(ENC_R_B) == HIGH) enc_right_count++;
  else                              enc_right_count--;
}

// ─────────────────────────────────────────────────────────────────────────────
// MOTOR CONTROL
// ─────────────────────────────────────────────────────────────────────────────
inline void setMotor(int channel, int dirPin, int speed) {
  speed = constrain(speed, -PWM_MAX, PWM_MAX);
  if (speed >= 0) { digitalWrite(dirPin, LOW);  ledcWrite(channel,  speed); }
  else            { digitalWrite(dirPin, HIGH); ledcWrite(channel, -speed); }
}

void driveAll(int left, int right) {
  setMotor(0, FL_DIR, left);  setMotor(2, ML_DIR, left);  setMotor(4, RL_DIR, left);
  setMotor(1, FR_DIR, right); setMotor(3, MR_DIR, right); setMotor(5, RR_DIR, right);
}

void stopAll() {
  driveAll(0, 0);
  cmd_linear_x = 0.0f;
  cmd_angular_z = 0.0f;
  L.integ = 0.0f; R.integ = 0.0f;
}

// Open-loop feedforward: velocity (m/s) → PWM with stall-floor MIN_PWM.
inline int vel_to_pwm(float vel_ms) {
  if (fabsf(vel_ms) < VEL_DEADBAND_MS) return 0;
  float sign  = (vel_ms > 0.0f) ? 1.0f : -1.0f;
  float ratio = fabsf(vel_ms) / MAX_SPEED_MS;
  float pwm   = MIN_PWM + (PWM_MAX - MIN_PWM) * constrain(ratio, 0.0f, 1.0f);
  return (int)(sign * pwm);
}

// PI correction around the feedforward, with anti-windup and a hard clamp.
inline int closed_loop_pwm(SideCtrl &s, float target_ms, float dt_s) {
  int ff = vel_to_pwm(target_ms);

#if CLOSED_LOOP
  if (fabsf(target_ms) < VEL_DEADBAND_MS) {   // stop request → no windup
    s.integ = 0.0f;
    return 0;
  }
  if (s.openloop_latch) return ff;            // encoder faulted → feedforward only

  float err = target_ms - s.meas_vel_ms;
  s.integ  += err * dt_s;
  s.integ   = constrain(s.integ, -I_CLAMP, I_CLAMP);
  float corr = KP_PWM_PER_MS * err + KI_PWM_PER_MS * s.integ;
  corr = constrain(corr, -CORR_CLAMP, CORR_CLAMP);

  return constrain(ff + (int)corr, -PWM_MAX, PWM_MAX);
#else
  return ff;
#endif
}

// Update one side's measured velocity, run the runaway guard, return PWM.
int control_side(SideCtrl &s, volatile long &counter, float enc_sign,
                 float target_ms, float dt_s) {
  noInterrupts();
  long now_c = counter;
  interrupts();
  long d = now_c - s.last_count;
  s.last_count = now_c;

  float inst_vel = enc_sign * (float)d * METERS_PER_COUNT / dt_s;
  s.meas_vel_ms += VEL_EMA_ALPHA * (inst_vel - s.meas_vel_ms);

  // Runaway guard: commanded hard but encoder dead or fighting the command.
  bool bad = (fabsf(target_ms) > GUARD_CMD_MS) &&
             (fabsf(s.meas_vel_ms) < GUARD_MIN_MEAS_MS ||
              (target_ms * s.meas_vel_ms) < 0.0f);
  if (bad) {
    s.fault_ms += (unsigned long)(dt_s * 1000.0f);
    if (s.fault_ms >= GUARD_MS && !s.openloop_latch) {
      s.openloop_latch = true;   // latch to feedforward-only for this side
      s.integ = 0.0f;
    }
  } else {
    s.fault_ms = 0;
  }

  return closed_loop_pwm(s, target_ms, dt_s);
}

// Convert /cmd_vel to per-side target speeds and drive with closed loop.
void drive_rover(float linear_x, float angular_z, float dt_s) {
  float left_target  = linear_x - (angular_z * TRACK_WIDTH_M / 2.0f);
  float right_target = linear_x + (angular_z * TRACK_WIDTH_M / 2.0f);

  int left_pwm  = control_side(L, enc_left_count,  ENC_L_SIGN, left_target,  dt_s);
  int right_pwm = control_side(R, enc_right_count, ENC_R_SIGN, right_target, dt_s);

  driveAll(left_pwm, right_pwm);
}

// ─────────────────────────────────────────────────────────────────────────────
// ROS2 CALLBACKS
// ─────────────────────────────────────────────────────────────────────────────
void cmd_vel_callback(const void *msgin) {
  const geometry_msgs__msg__Twist *msg = (const geometry_msgs__msg__Twist *)msgin;
  cmd_linear_x  = msg->linear.x;
  cmd_angular_z = msg->angular.z;
  last_cmd_vel_time = millis();
}

// ─────────────────────────────────────────────────────────────────────────────
// micro-ROS LIFECYCLE
// ─────────────────────────────────────────────────────────────────────────────
bool create_entities() {
  allocator = rcl_get_default_allocator();
  if (rclc_support_init(&support, 0, NULL, &allocator) != RCL_RET_OK) return false;
  if (rclc_node_init_default(&node, "rover_6wd", "", &support) != RCL_RET_OK) return false;

  if (rclc_subscription_init_default(
        &cmd_vel_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
        "/cmd_vel") != RCL_RET_OK) return false;

  if (rclc_publisher_init_default(
        &rpm_pub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        "/rover/rpm") != RCL_RET_OK) return false;

  rpm_msg.data.data     = rpm_data;
  rpm_msg.data.size     = 4;
  rpm_msg.data.capacity = 4;

  if (rclc_executor_init(&executor, &support.context, 1, &allocator) != RCL_RET_OK) return false;
  if (rclc_executor_add_subscription(
        &executor, &cmd_vel_sub, &cmd_vel_msg,
        &cmd_vel_callback, ON_NEW_DATA) != RCL_RET_OK) return false;

  // Reset control + odometry snapshots so no stale delta on reconnect.
  noInterrupts();
  L.last_count = enc_left_count;   R.last_count = enc_right_count;
  pub_last_left = enc_left_count;  pub_last_right = enc_right_count;
  interrupts();
  L.meas_vel_ms = R.meas_vel_ms = 0.0f;
  L.integ = R.integ = 0.0f;
  L.openloop_latch = R.openloop_latch = false;
  L.fault_ms = R.fault_ms = 0;

  last_publish_time = millis();
  last_cmd_vel_time = 0;
  return true;
}

void destroy_entities() {
  rmw_context_t *rmw_ctx = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_ctx, 0);
  rcl_publisher_fini(&rpm_pub, &node);
  rcl_subscription_fini(&cmd_vel_sub, &node);
  rclc_executor_fini(&executor);
  rcl_node_fini(&node);
  rclc_support_fini(&support);
}

// ─────────────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  // Serial is the micro-ROS XRCE-DDS transport (set_microros_transports below).
  // No plain-text Serial.print() past that call — it corrupts the agent link.
  Serial.begin(115200);
  delay(1000);

  // Motor direction pins
  pinMode(FL_DIR, OUTPUT); pinMode(FR_DIR, OUTPUT);
  pinMode(ML_DIR, OUTPUT); pinMode(MR_DIR, OUTPUT);
  pinMode(RL_DIR, OUTPUT); pinMode(RR_DIR, OUTPUT);

  // PWM channels
  ledcSetup(0, PWM_FREQ, PWM_RES); ledcSetup(1, PWM_FREQ, PWM_RES);
  ledcSetup(2, PWM_FREQ, PWM_RES); ledcSetup(3, PWM_FREQ, PWM_RES);
  ledcSetup(4, PWM_FREQ, PWM_RES); ledcSetup(5, PWM_FREQ, PWM_RES);

  ledcAttachPin(FL_PWM, 0); ledcAttachPin(FR_PWM, 1);
  ledcAttachPin(ML_PWM, 2); ledcAttachPin(MR_PWM, 3);
  ledcAttachPin(RL_PWM, 4); ledcAttachPin(RR_PWM, 5);

  stopAll();  // motors off immediately

  // Encoder pins (34/35/36/39 are input-only — no pullMode; add external
  // pull-ups if the encoder outputs are open-collector).
  pinMode(ENC_L_A, INPUT); pinMode(ENC_L_B, INPUT);
  pinMode(ENC_R_A, INPUT); pinMode(ENC_R_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), enc_left_isr,  RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), enc_right_isr, RISING);

  // BNO055 IMU (I2C, separate bus from Serial -- safe to init before or
  // after set_microros_transports() below; done here so imu_present is
  // known before the first RPM publish). No effect on wheel control if the
  // sensor isn't wired/found: imu_present just stays false and
  // bno055_read_heading_deg() reports NaN forever, same as no IMU at all.
  imu_present = bno055_init();

  // After this, Serial belongs to the micro-ROS transport. No Serial.print().
  set_microros_transports();
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN LOOP — reconnection state machine + fixed-rate control loop
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  switch (agent_state) {

    case WAITING_AGENT:
      stopAll();
      if (rmw_uros_ping_agent(AGENT_PING_TIMEOUT_MS, AGENT_PING_ATTEMPTS) == RMW_RET_OK) {
        if (create_entities()) {
          agent_state = AGENT_CONNECTED;
        } else {
          destroy_entities();
          delay(500);
        }
      } else {
        delay(500);
      }
      break;

    case AGENT_CONNECTED: {
      rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

      // Low-frequency agent liveness check (every 2 s)
      static unsigned long last_ping_time = 0;
      if (millis() - last_ping_time >= 2000) {
        last_ping_time = millis();
        if (rmw_uros_ping_agent(100, 1) != RMW_RET_OK) {
          agent_state = AGENT_DISCONNECTED;
          break;
        }
      }

      // ── Fixed-rate closed-loop control tick ─────────────────────────────
      static unsigned long last_control_time = 0;
      unsigned long now = millis();
      if (now - last_control_time >= CONTROL_DT_MS) {
        float dt_s = (now - last_control_time) * 0.001f;
        last_control_time = now;

        // cmd_vel watchdog: zero on silence
        if (last_cmd_vel_time == 0 ||
            (now - last_cmd_vel_time > CMD_VEL_TIMEOUT_MS)) {
          cmd_linear_x = 0.0f; cmd_angular_z = 0.0f;
        }
        drive_rover(cmd_linear_x, cmd_angular_z, dt_s);
      }

      // ── 10 Hz signed RPM publish (odometry contract) ────────────────────
      if (now - last_publish_time >= RPM_PUBLISH_MS) {
        float dt_min = (now - last_publish_time) / 60000.0f;  // ms → minutes

        noInterrupts();
        long lc = enc_left_count;
        long rc = enc_right_count;
        interrupts();

        long dl = lc - pub_last_left;   pub_last_left  = lc;
        long dr = rc - pub_last_right;  pub_last_right = rc;

        // signed RPM at the wheel (+ve = drives robot forward)
        rpm_data[0] = ENC_L_SIGN * (dl / ENCODER_CPR) / dt_min;
        rpm_data[1] = ENC_R_SIGN * (dr / ENCODER_CPR) / dt_min;
        rpm_data[2] = bno055_read_heading_deg();  // NaN if IMU absent/unread
        {
          uint8_t c = bno055_read_calib_status();
          float sys_c = (c >> 6) & 0x03, gyr_c = (c >> 4) & 0x03,
                acc_c = (c >> 2) & 0x03, mag_c = c & 0x03;
#if TRUST_LOADED_MAG_CALIB
          // See TRUST_LOADED_MAG_CALIB's comment above: bridge only up to
          // the first genuine live confirmation this boot, then get out of
          // the way permanently so a later real disturbance isn't masked.
          if (mag_c >= 3.0f) bno_mag_live_confirmed = true;
          if (bno_calib_loaded_this_boot && !bno_mag_live_confirmed && mag_c < 3.0f) {
            mag_c = 3.0f;
          }
#endif
          rpm_data[3] = sys_c * 1000.0f + gyr_c * 100.0f + acc_c * 10.0f + mag_c;
          // Auto-persist the calibration offsets the FIRST time full
          // calibration (all four sub-scores == 3, i.e. c == 0xFF) is seen
          // this boot -- see bno055_save_calibration()'s docstring. After
          // this, a power cycle restores it via bno055_load_calibration()
          // in bno055_init() instead of needing the manual wave-around
          // dance again.
          if (!bno_calib_saved_this_boot && c == 0xFF) {
            bno055_save_calibration();
            bno_calib_saved_this_boot = true;
          }
        }
        rcl_publish(&rpm_pub, &rpm_msg, NULL);

        // NOTE: Serial is the XRCE-DDS transport — do NOT Serial.print() here.
        last_publish_time = now;
      }

      delay(2);
      break;
    }

    case AGENT_DISCONNECTED:
      stopAll();
      destroy_entities();
      agent_state = WAITING_AGENT;
      break;
  }
}

/*
 * FLASHING  (new ESP32 — first flash)
 * ───────────────────────────────────
 * Board  : ESP32 Dev Module          Speed : 115200          Port : /dev/ttyUSB0 (Pi)
 * Library: micro_ros_arduino (Humble)
 *
 * Compile on the GPU host (offline toolchain), flash from the Pi with esptool.
 * Kill the micro-ROS respawn wrappers first (see esp32-flashing-procedure memo).
 *
 * After flash, start agent on Pi:
 *   ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
 *
 * BENCH CHECKLIST (do before driving autonomously):
 *   1. ros2 topic echo /rover/rpm — spin each wheel FORWARD by hand; RPM must
 *      read POSITIVE. If a side is negative, flip ENC_L_SIGN / ENC_R_SIGN.
 *   2. Verify ENCODER_CPR: push the rover exactly one wheel revolution, confirm
 *      the integrated count ≈ ENCODER_CPR. Adjust if there is a gearbox.
 *   3. Command a slow forward and confirm both wheels hold speed (closed loop).
 *      Set CLOSED_LOOP 0 to fall back to open-loop feedforward if needed.
 *   4. ros2 topic echo /rover/rpm — a 3rd array element should now appear.
 *      A real number (not NaN) confirms the BNO055 wiring (VIN/GND/SDA/SCL)
 *      is correct and the sensor acked its CHIP_ID at boot; NaN means
 *      bno055_init() never saw the sensor -- check wiring before trusting
 *      any heading-based feature built on this field.
 */
