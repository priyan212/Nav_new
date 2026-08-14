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
 *                             imu_heading_deg: BNO08x fused heading (yaw
 *                             from the GAME Rotation Vector report -- see
 *                             2026-08-14 addendum below for why not the
 *                             plain Rotation Vector) in degrees [0,360), or
 *                             NaN if the IMU never produced a report;
 *                             imu_calib: the GYROSCOPE's own 0-3 accuracy
 *                             status (see addendum) broadcast into ALL FOUR
 *                             digits (SYS=GYR=ACC=MAG=status, e.g. status 3
 *                             -> 3333) so existing consumers that gate on
 *                             the MAG/ones digit (odometry_logger.py et al,
 *                             unchanged, see --imu-min-mag-calib) keep
 *                             working unmodified -- unlike the BNO055 this
 *                             replaces, the BNO08x's SH-2 firmware reports
 *                             per-REPORT accuracy, not simultaneous
 *                             independent SYS/GYR/ACC/MAG sub-scores from
 *                             one report, so those four digits are no
 *                             longer independent, just replicated for wire
 *                             compatibility.
 *
 * IMU (2026-08-14, replaces an earlier BNO055): Adafruit BNO085/BNO080
 * ("9-DOF Orientation IMU Fusion Breakout", STEMMA QT) on I2C (GPIO21 SDA /
 * GPIO22 SCL, ESP32 defaults -- same pins the BNO055 used, free on this
 * board). Wheel-differential heading (theta) drifts sharply past
 * ~135-165deg of rotation on this skid-steer chassis (measured, see
 * odometry_log/odom_accuracy_results.csv) because turning scrubs the
 * wheels against the ground -- the encoders can't see that slip. The
 * BNO08x does its own onboard sensor fusion (accel+gyro+mag, CEVA SH-2
 * firmware) and reports an absolute heading that doesn't depend on wheel
 * odometry at all, so it's immune to that specific failure mode -- same
 * motivation as the BNO055 before it.
 *
 * UNLIKE the BNO055 (simple flat I2C registers), the BNO08x speaks SHTP, a
 * packetized sensor-hub protocol -- there's no raw-register shortcut, so
 * this firmware uses Adafruit's official Adafruit_BNO08x library (+ its
 * Adafruit_BusIO / Adafruit_Sensor deps, + the bundled CEVA `sh2` C driver)
 * instead of hand-rolled I2C like the BNO055 code did. See
 * bno08x_init()/bno08x_poll() below. Calibration persistence works
 * differently too and is actually SIMPLER: the BNO08x's own SH-2 firmware
 * owns Dynamic Calibration Data (DCD) in its own internal flash --
 * sh2_setDcdAutoSave(true) in bno08x_init() is a one-line fire-and-forget
 * equivalent of the BNO055 code's entire NVS save/restore dance (no
 * ESP32-side Preferences blob, no host-side load-at-boot step). Getting
 * CALIBRATED (Pi-side theta_src == "imu") does NOT mean the heading is
 * actually ACCURATE, though: --imu-min-mag-calib on the Pi side
 * (home_gui.py et al) gates on the imu_calib ones digit, and level 1 there
 * (the default) is a low bar -- real magnetic disturbance near the sensor
 * (motor current, nearby metal/wiring) can still swing the reported heading
 * meaningfully even at status==1. Status 3 is this chip's own bar for a
 * trustworthy absolute heading; if it can't reach 3 in a given room, that's
 * a real environment/mounting issue, not a firmware gap.
 *
 * BENCH-VALIDATED 2026-08-14, TWO PASSES: first pass commanded
 * angular_z=+0.25 rad/s (ROS CCW convention) for 2s over live cmd_vel and
 * found a smooth, monotonic heading INCREASE (98.4deg -> 136.6deg) -- correct
 * direction relative to this firmware's own encoder sign convention, but
 * that's ROS CCW+, NOT the "compass CW+" odometry_logger.py's _imu_theta()
 * actually assumes (see bno08x_heading_from_quat()'s comment) -- caught
 * during a later doc-consistency pass, not the original bench test. Fixed
 * by negating the yaw there; reflashed and repeated the spin (angular_z=
 * +0.5 for 3s, left_rpm ~-31 / right_rpm ~+30 confirming a real CCW
 * rotation) -- heading now DECREASED smoothly and monotonically
 * (271.8deg -> 175.4deg, ~96deg, no jumps) for the same +angular_z (CCW)
 * command, i.e. now behaves like the old BNO055 the Pi side was written
 * against. Zero-offset (what heading reads while facing any particular
 * real-world direction) was
 * never calibrated against a compass/landmark -- only sign/direction.
 *
 * ADDENDUM 2026-08-14 -- switched Rotation Vector -> GAME Rotation Vector.
 * The plain (9-DOF, magnetometer-referenced) Rotation Vector's accuracy
 * status was found STUCK AT 0 ("Unreliable") through 700+ degrees of real
 * commanded rotation (both directions) AND a 20s stationary window --
 * confirmed via a standalone diagnostic sketch that GYR reached status 3
 * and ACC reached status 2 in the same session (so the calibration engine
 * itself works), while MAG's raw field readings were physically plausible
 * (~24-25uT, real Earth-field magnitude, not zero/garbage) but its
 * calibration CONFIDENCE never rose regardless -- pointing at magnetic
 * interference near the mount (motors/wiring/chassis metal) rather than a
 * broken sensor or a failed sh2_setCalConfig() call. This matches
 * independent reports from BNO08x users elsewhere: the 9-DOF Rotation
 * Vector's magnetometer fusion is widely described as fragile/opaque near
 * motors, and Game Rotation Vector (accel+gyro only, NO magnetometer) is
 * the documented workaround specifically for robots with nearby motors
 * (CEVA/Adafruit report-type docs; see also forum.arduino.cc's BNO085
 * calibration thread). Tradeoff: Game RV has no absolute magnetic-north
 * reference, so it WILL drift slowly over very long sessions, unlike a
 * genuinely-calibrated Rotation Vector -- but that's the honest choice
 * here, since the magnetometer-referenced path was never actually reaching
 * calibrated on this rover, meaning it was contributing NOTHING (odometry
 * always fell back to wheel-diff) while still carrying the interference
 * risk. Gyro drift over a typical single-goal session (minutes, not hours)
 * should be far smaller than the wheel-diff drift it replaces (see the
 * ~135-165deg wheel-diff failure mode described above) -- not yet
 * long-session bench-validated, though. imu_calib's gating digit now comes
 * from the GYROSCOPE's own status (SH2_GYROSCOPE_CALIBRATED report,
 * confirmed reaching 3 reliably and fast in the same diagnostic session)
 * instead of the magnetometer's, since gyro (not mag) is what Game RV's
 * quality actually depends on.
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
#include <Adafruit_BNO08x.h>  // BNO08x SHTP driver (+ Adafruit_BusIO / Adafruit_Sensor
                               // deps, + bundled CEVA sh2 C driver) -- see IMU note above
                               // for why this needed a real library, unlike the BNO055
                               // it replaces.

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
// BNO08x IMU (I2C, absolute-heading fusion via SHTP/SH-2) — see the file
// header note above for why this needs the Adafruit_BNO08x library instead
// of the BNO055's hand-rolled raw registers.
// ─────────────────────────────────────────────────────────────────────────────
#define BNO08X_I2C_ADDR       BNO08x_I2CADDR_DEFAULT  // 0x4A (ADR jumper solders to 0x4B --
                                                       // bno08x_init() tries both, see below)
#define BNO08X_I2C_ADDR_ALT   0x4B
#define BNO08X_SDA_PIN        21
#define BNO08X_SCL_PIN        22
#define BNO08X_REPORT_US      20000  // ask the sensor for Game Rotation Vector / Gyroscope
                                      // reports every 20ms (50Hz) each -- polled by
                                      // bno08x_poll() every loop() iteration (well under
                                      // 50Hz) and only READ (not requested) by the 10Hz
                                      // publisher, so this just needs to stay comfortably
                                      // faster than RPM_PUBLISH_MS.

Adafruit_BNO08x bno08x;
sh2_SensorValue_t bno08x_value;

bool  imu_present            = false;
float imu_heading_deg_cached = NAN;   // updated by bno08x_poll(), read at publish time
uint8_t imu_accuracy_cached  = 0;     // 0(unreliable)-3(high), the GAME Rotation Vector
                                       // report's own accuracy status (accel+gyro derived,
                                       // no magnetometer -- see file header's 2026-08-14
                                       // addendum for the full story of why this changed from
                                       // the plain Rotation Vector). Also broadcast into all
                                       // four rpm_data[3] digits -- see gyro_accuracy_cached
                                       // below for what specifically gates theta trust.
uint8_t gyro_accuracy_cached = 0;     // 0-3, the GYROSCOPE'S OWN status from a separate
                                       // SH2_GYROSCOPE_CALIBRATED report -- THIS is what maps
                                       // to rpm_data[3]'s ones/MAG digit and what
                                       // --imu-min-mag-calib actually gates on Pi-side (name
                                       // unchanged Pi-side; it now really means "IMU min
                                       // gyro calib" but the flag/wire contract stayed put on
                                       // purpose -- see file header addendum). Confirmed via a
                                       // standalone diagnostic sketch to reach 3 reliably and
                                       // fast, unlike the magnetometer's status this replaced,
                                       // which never left 0 on this rover.

// Quaternion (i,j,k,real) -> yaw heading, degrees [0,360), COMPASS
// CONVENTION (CW+, i.e. increases while turning clockwise viewed from
// above) -- NOT the ROS-standard CCW+ a bare atan2 yaw extraction gives.
// This matters: odometry_logger.py's _imu_theta() explicitly assumes
// "compass CW+" input (see its own comment, `delta_deg = ref - current
// # compass CW+ -> theta CCW+") because that's what the BNO055 this
// firmware replaced reported via its Euler-heading register (Bosch's
// datasheet convention, matches a real magnetic compass). Bench-tested
// 2026-08-14 (see file header): the RAW aerospace atan2(2*(w*z+x*y),
// 1-2*(y^2+z^2)) formula, WITHOUT the negation below, increases with CCW
// rotation (verified: +angular_z spin increased it) -- i.e. it's the
// opposite convention from what the Pi side expects. Negated here so
// imu_heading_deg keeps behaving exactly like the old BNO055's did and
// odometry_logger.py needs ZERO changes.
float bno08x_heading_from_quat(float i, float j, float k, float real) {
  float siny_cosp = 2.0f * (real * k + i * j);
  float cosy_cosp = 1.0f - 2.0f * (j * j + k * k);
  float yaw_deg = -atan2f(siny_cosp, cosy_cosp) * (180.0f / PI);  // negate: CCW+ -> CW+
  if (yaw_deg < 0.0f) yaw_deg += 360.0f;
  return yaw_deg;
}

// Returns false if the BNO08x never identifies/ack's correctly at EITHER
// candidate address, or the Game Rotation Vector report can't be enabled --
// this IS the wiring check, same role bno055_init()'s old CHIP_ID read
// played. Tries the default address first, then the ADR-jumper-soldered
// alternate (STEMMA QT breakouts commonly ship with this jumper in either
// state) -- if BOTH fail, that's a real wiring/power problem, not an
// address guess.
bool bno08x_init() {
  Wire.begin(BNO08X_SDA_PIN, BNO08X_SCL_PIN);

  if (!bno08x.begin_I2C(BNO08X_I2C_ADDR, &Wire) &&
      !bno08x.begin_I2C(BNO08X_I2C_ADDR_ALT, &Wire)) {
    return false;
  }

  // Let the chip's own SH-2 firmware own calibration persistence in ITS
  // flash, not the ESP32's -- replaces the BNO055 code's entire NVS
  // save/restore dance with two fire-and-forget calls (see file header).
  // SH2_CAL_MAG left enabled even though Game Rotation Vector doesn't use
  // the magnetometer -- harmless, and keeps the door open if a future
  // remount away from motor interference makes the plain Rotation Vector
  // viable again (see 2026-08-14 addendum).
  sh2_setCalConfig(SH2_CAL_ACCEL | SH2_CAL_GYRO | SH2_CAL_MAG);
  sh2_setDcdAutoSave(true);

  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, BNO08X_REPORT_US)) return false;
  // Best-effort: if the gyro-specific report can't be enabled for some
  // reason, gyro_accuracy_cached just stays 0 (same as "never calibrated")
  // rather than failing bno08x_init() entirely -- heading still works fine
  // off the Game Rotation Vector report alone either way.
  bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, BNO08X_REPORT_US);
  return true;
}

// Drain any pending SHTP report(s) and cache the latest heading/accuracy --
// call every loop() iteration (cheap; non-blocking if nothing is pending).
// Also handles a BNO08x-side reset (e.g. a brownout on flexed wiring) by
// re-enabling the report -- unlike the BNO055, which needed a full ESP32
// reboot to recover from that (see esp32-flashing-procedure memo), the
// BNO08x can self-heal mid-session here without one.
void bno08x_poll() {
  if (!imu_present) return;

  if (bno08x.wasReset()) {
    bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, BNO08X_REPORT_US);
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, BNO08X_REPORT_US);
  }

  // getSensorEvent() returns (up to) one report per call, whichever's next
  // in the SHTP queue -- checking sensorId routes it to the right cache.
  // Called every loop() iteration, comfortably faster than either report's
  // 20ms interval, so neither queue backs up.
  if (bno08x.getSensorEvent(&bno08x_value)) {
    if (bno08x_value.sensorId == SH2_GAME_ROTATION_VECTOR) {
      imu_heading_deg_cached = bno08x_heading_from_quat(
          bno08x_value.un.gameRotationVector.i, bno08x_value.un.gameRotationVector.j,
          bno08x_value.un.gameRotationVector.k, bno08x_value.un.gameRotationVector.real);
      imu_accuracy_cached = bno08x_value.status & 0x03;  // bits 1-0: accuracy, see sh2_SensorValue.h
    } else if (bno08x_value.sensorId == SH2_GYROSCOPE_CALIBRATED) {
      gyro_accuracy_cached = bno08x_value.status & 0x03;
    }
  }
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

  // BNO08x IMU (I2C, separate bus from Serial -- safe to init before or
  // after set_microros_transports() below; done here so imu_present is
  // known before the first RPM publish). No effect on wheel control if the
  // sensor isn't wired/found: imu_present just stays false and
  // imu_heading_deg_cached stays NaN forever, same as no IMU at all.
  imu_present = bno08x_init();

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

      // Drain any pending BNO08x report every iteration (cheap) so
      // imu_heading_deg_cached/imu_accuracy_cached are fresh whenever the
      // 10 Hz publisher below reads them.
      bno08x_poll();

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
        rpm_data[2] = imu_heading_deg_cached;  // NaN if IMU absent/never reported
        // SYS/GYR/ACC digits (thousands/hundreds/tens) all carry the Game
        // Rotation Vector's own accel+gyro-derived accuracy status --
        // informational only, nothing currently gates on them. The
        // ones/MAG digit carries the GYROSCOPE'S OWN independent status
        // instead (see gyro_accuracy_cached's comment above for why --
        // switched from magnetometer 2026-08-14, see file header) -- that's
        // the digit --imu-min-mag-calib actually gates theta-trust on,
        // Pi-side.
        rpm_data[3] = imu_accuracy_cached * 1110.0f + gyro_accuracy_cached;
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
 * Library: micro_ros_arduino (Humble), Adafruit_BNO08x + Adafruit_BusIO +
 *          Adafruit_Sensor (cloned straight from github.com/adafruit into
 *          ~/Arduino/libraries/ on the GPU host, 2026-08-14 — no offline
 *          package/version pin exists for these yet, unlike micro_ros_arduino;
 *          if a future recompile behaves differently, check these libraries'
 *          installed commit first).
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
 *      A real number (not NaN) confirms the BNO08x wiring (VIN/GND/SDA/SCL)
 *      is correct and begin_I2C()/enableReport() succeeded at boot; NaN means
 *      bno08x_init() never saw the sensor, OR it's wired/addressed correctly
 *      but the ADR jumper is soldered (chip is actually answering at 0x4B,
 *      not the BNO08X_I2C_ADDR default of 0x4A) -- check wiring AND the
 *      jumper before trusting any heading-based feature built on this field.
 *   5. DONE 2026-08-14 — heading sign validated live over cmd_vel/rover/rpm,
 *      two passes (first pass caught a real CCW-vs-CW convention mismatch
 *      against the Pi side, fixed with a negation — see
 *      bno08x_heading_from_quat()'s comment and the file header for the
 *      full story). Final confirmed behavior: commanded angular_z=+0.5 for
 *      3s (left_rpm ~-31 / right_rpm ~+30, a real CCW rotation) produced a
 *      smooth, monotonic heading DECREASE (271.8deg -> 175.4deg) with no
 *      jumps/reversal — matches the old BNO055's compass CW+ convention.
 *      Absolute zero-offset (vs. a compass/landmark) was NOT checked.
 */
