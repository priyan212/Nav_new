#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <geometry_msgs/msg/twist.h>
#include <nav_msgs/msg/odometry.h>
#include <std_msgs/msg/float32_multi_array.h>

#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <math.h>

// ─────────────────────────────────────────────────────────────────────────────
//  MOTOR PINS
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
//  ENCODER PINS
//  GPIO 34-39 are input-only on ESP32 — no internal pull-up available.
//  Wire external 4.7kΩ resistors from each pin to 3.3V (not 5V).
//  4.7kΩ gives cleaner rising/falling edges than 10kΩ at high RPM.
// ─────────────────────────────────────────────────────────────────────────────
#define ENC_L_A 36
#define ENC_L_B 39
#define ENC_R_A 34
#define ENC_R_B 35

// ─────────────────────────────────────────────────────────────────────────────
//  ODOMETRY CONSTANTS
//  WHEEL_BASE_M is the single most important constant for heading accuracy.
//  Measure the distance between the two encoder wheel contact patches,
//  not the outer frame — it is usually a few mm narrower than it looks.
// ─────────────────────────────────────────────────────────────────────────────
#define TICKS_PER_REV     2400.0f   // x4 quadrature × 600 CPR encoder
#define WHEEL_DIAMETER_M  0.112f
#define WHEEL_BASE_M      0.263f    // ← measure and update this precisely
#define WHEEL_CIRC        (M_PI * WHEEL_DIAMETER_M)
#define DIST_PER_TICK     (WHEEL_CIRC / TICKS_PER_REV)

// ─────────────────────────────────────────────────────────────────────────────
//  PWM SETTINGS
// ─────────────────────────────────────────────────────────────────────────────
#define PWM_FREQ   20000
#define PWM_RES    10
#define PWM_MAX    1023
#define MIN_PWM    450      // overcome static friction — tune per motor set

// ─────────────────────────────────────────────────────────────────────────────
//  TIMING
// ─────────────────────────────────────────────────────────────────────────────
#define CONTROL_DT_MS   10      // 100 Hz control + odometry loop
#define CMD_TIMEOUT_MS  800     // stop if no cmd_vel — large enough for serial lag

// PING_RETRIES must be high because serial_read is non-blocking (returns 0
// immediately if the buffer is empty).  micro-ROS loops calling serial_read
// rapidly within each PING_TIMEOUT_MS window; more retries = more total time
// for the Pi agent's response to arrive.  20 × 500 ms = 10 s total window.
#define PING_TIMEOUT_MS 500
#define PING_RETRIES    20

// ─────────────────────────────────────────────────────────────────────────────
//  VELOCITY EMA FILTER
//  α = 0.3 → ~3-tick time constant at 100 Hz = 30 ms effective lag.
// ─────────────────────────────────────────────────────────────────────────────
#define VEL_ALPHA    0.3f

// ─────────────────────────────────────────────────────────────────────────────
//  STEER EMA FILTER
//  α = 0.35 → ~65 ms time constant at 100 Hz control loop.
//  Provides a second smoothing layer after the Python-side slew-rate limiter
//  so any residual step changes in steerPct are further damped before the
//  wheels feel them.  Raise α toward 1.0 for faster response.
// ─────────────────────────────────────────────────────────────────────────────
#define STEER_ALPHA  0.35f

// ─────────────────────────────────────────────────────────────────────────────
//  DRIFT CORRECTION
//  Slow integrator — no PID pulsing.
//  DRIFT_KI = 0.3 → ~3 s to build full DRIFT_MAX correction.
//  Only active on straight lines; decays during turns.
// ─────────────────────────────────────────────────────────────────────────────
#define DRIFT_KI   0.3f
#define DRIFT_MAX  80.0f

// ─────────────────────────────────────────────────────────────────────────────
//  SPEED SCALE
// ─────────────────────────────────────────────────────────────────────────────
#define MAX_SPEED_MS  0.5f    // m/s at full cmd_vel throttle

// ─────────────────────────────────────────────────────────────────────────────
//  AUTO-RECONNECT STATE MACHINE
// ─────────────────────────────────────────────────────────────────────────────
enum AgentState { WAITING_AGENT, AGENT_CONNECTED, AGENT_DISCONNECTED };
AgentState agentState = WAITING_AGENT;

// ─────────────────────────────────────────────────────────────────────────────
//  ENCODER — x4 quadrature lookup table
//  QEM[prevAB << 2 | curAB] → +1 forward, -1 backward, 0 glitch/no-move
// ─────────────────────────────────────────────────────────────────────────────
static const int8_t QEM[16] = {
  0, -1, +1,  0,
  +1,  0,  0, -1,
  -1,  0,  0, +1,
  0, +1, -1,  0
};

volatile long    encCountL = 0, encCountR = 0;
volatile uint8_t encStateL = 0, encStateR = 0;
long prevEncL = 0, prevEncR = 0;

float velL_filt = 0.0f, velR_filt = 0.0f;
float odom_x = 0.0f, odom_y = 0.0f, odom_theta = 0.0f;
float driftBias = 0.0f;

// ─────────────────────────────────────────────────────────────────────────────
//  ENCODER ISRs
// ─────────────────────────────────────────────────────────────────────────────
void IRAM_ATTR isrEncL() {
  uint8_t cur = (digitalRead(ENC_L_A) << 1) | digitalRead(ENC_L_B);
  encCountL  += QEM[(encStateL << 2) | cur];
  encStateL   = cur;
}

void IRAM_ATTR isrEncR() {
  uint8_t cur = (digitalRead(ENC_R_A) << 1) | digitalRead(ENC_R_B);
  encCountR  += QEM[(encStateR << 2) | cur];
  encStateR   = cur;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ROS VARIABLES
// ─────────────────────────────────────────────────────────────────────────────
rcl_node_t          node;
rclc_support_t      support;
rcl_allocator_t     allocator;
rclc_executor_t     executor;
rcl_subscription_t  sub_cmdvel;
geometry_msgs__msg__Twist cmd_msg;
rcl_publisher_t     pub_odom;
nav_msgs__msg__Odometry odom_msg;
rcl_publisher_t     pub_speeds;
std_msgs__msg__Float32MultiArray speeds_msg;
float               speeds_data[2];
rcl_timer_t         timer;

// Frame IDs — must persist for the lifetime of the node (static storage)
static char ODOM_FRAME[]      = "odom";
static char BASE_LINK_FRAME[] = "base_link";

// ─────────────────────────────────────────────────────────────────────────────
//  CONTROL STATE
// ─────────────────────────────────────────────────────────────────────────────
float targetLeft  = 0.0f;
float targetRight = 0.0f;
float steerFilt   = 0.0f;   // EMA-smoothed steer fraction (–1..1)
volatile unsigned long lastCmdTime = 0;

// ─────────────────────────────────────────────────────────────────────────────
//  SERIAL TRANSPORT — non-blocking
//  serial_read returns only what is already buffered — never blocks on timeout.
//  This stops the executor from stalling between messages, but requires
//  PING_RETRIES to be high so the ping loop has enough time to collect
//  the agent's response bytes across multiple rapid read calls.
// ─────────────────────────────────────────────────────────────────────────────
bool serial_open (struct uxrCustomTransport*) { return true; }
bool serial_close(struct uxrCustomTransport*) { return true; }

size_t serial_write(struct uxrCustomTransport*,
                    const uint8_t* buf, size_t len, uint8_t*) {
  Serial.write(buf, len);
  return len;
}

size_t serial_read(struct uxrCustomTransport*,
                  uint8_t* buf, size_t len, int /*timeout*/, uint8_t*) {
  size_t avail = Serial.available();
  if (avail == 0) return 0;
  return Serial.readBytes((char*)buf, min(len, avail));
}

// ─────────────────────────────────────────────────────────────────────────────
//  MOTOR HELPERS
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
  targetLeft  = 0.0f;
  targetRight = 0.0f;
  steerFilt   = 0.0f;   // clear steer EMA so it doesn't linger after e-stop
  driftBias   = 0.0f;
  velL_filt   = 0.0f;
  velR_filt   = 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  VELOCITY → PWM  (feedforward only)
// ─────────────────────────────────────────────────────────────────────────────
inline int vel_to_pwm(float vel_ms) {
  if (fabsf(vel_ms) < 0.05f) return 0;
  float sign  = (vel_ms > 0.0f) ? 1.0f : -1.0f;
  float ratio = fabsf(vel_ms) / MAX_SPEED_MS;
  float pwm   = MIN_PWM + (PWM_MAX - MIN_PWM) * constrain(ratio, 0.0f, 1.0f);
  return (int)(sign * pwm);
}

// ─────────────────────────────────────────────────────────────────────────────
//  CMD_VEL CALLBACK
// ─────────────────────────────────────────────────────────────────────────────
// MAX_ANGULAR_RAD must match max_angular in the Python PD controller.
// Python sends angular.z up to ±MAX_ANGULAR_RAD; normalising by the same
// value maps the full Python output range → full ±1.0 wheel differential.
// Previously 1.0 was used here, which clipped effective steer at 80% when
// max_angular was 0.8 — the rover went (mostly) straight.
#define MAX_ANGULAR_RAD  1.2f

void cmdVelCallback(const void* msgin) {
  const geometry_msgs__msg__Twist* msg =
      (const geometry_msgs__msg__Twist*)msgin;
  lastCmdTime = millis();

  float speedPct    = constrain(msg->linear.x  / MAX_SPEED_MS,    -1.0f, 1.0f);
  float steerRaw    = constrain(msg->angular.z / MAX_ANGULAR_RAD, -1.0f, 1.0f);

  // EMA low-pass: blend new steer command toward previous value.
  // This damps any residual step changes arriving from the network / Python
  // after the Python-side slew-rate limiter.
  steerFilt = STEER_ALPHA * steerRaw + (1.0f - STEER_ALPHA) * steerFilt;

  float leftPct  = constrain(speedPct - steerFilt, -1.0f, 1.0f);
  float rightPct = constrain(speedPct + steerFilt, -1.0f, 1.0f);

  targetLeft  = (fabsf(leftPct)  < 0.05f) ? 0.0f : leftPct  * MAX_SPEED_MS;
  targetRight = (fabsf(rightPct) < 0.05f) ? 0.0f : rightPct * MAX_SPEED_MS;
}

// ─────────────────────────────────────────────────────────────────────────────
//  TIMER CALLBACK — 100 Hz
// ─────────────────────────────────────────────────────────────────────────────
void timer_callback(rcl_timer_t*, int64_t) {
  const float dt = CONTROL_DT_MS / 1000.0f;

  // ── Watchdog ──────────────────────────────────────────────────────────────
  if ((millis() - lastCmdTime) > CMD_TIMEOUT_MS) {
    stopAll();
    speeds_data[0] = 0.0f; speeds_data[1] = 0.0f;
    speeds_msg.data.data = speeds_data;
    speeds_msg.data.size = speeds_msg.data.capacity = 2;
    rcl_publish(&pub_speeds, &speeds_msg, NULL);
    return;
  }

  // ── Atomic encoder snapshot ───────────────────────────────────────────────
  long curL, curR;
  noInterrupts();
  curL = encCountL;
  curR = encCountR;
  interrupts();

  long deltaL = curL - prevEncL;
  long deltaR = curR - prevEncR;
  prevEncL = curL;
  prevEncR = curR;

  // ── Raw wheel velocities (m/s) ────────────────────────────────────────────
  float velL_raw = (deltaL * DIST_PER_TICK) / dt;
  float velR_raw = (deltaR * DIST_PER_TICK) / dt;

  // ── EMA velocity filter ───────────────────────────────────────────────────
  velL_filt = VEL_ALPHA * velL_raw + (1.0f - VEL_ALPHA) * velL_filt;
  velR_filt = VEL_ALPHA * velR_raw + (1.0f - VEL_ALPHA) * velR_filt;

  // ── Exact arc (ICC) odometry ──────────────────────────────────────────────
  float distL  = deltaL * DIST_PER_TICK;
  float distR  = deltaR * DIST_PER_TICK;
  float dTheta = (distR - distL) / WHEEL_BASE_M;
  float theta0 = odom_theta;
  float theta1 = theta0 + dTheta;

  if (fabsf(dTheta) > 1e-6f) {
    float r  = (distL + distR) / (2.0f * dTheta);
    odom_x  += r * (sinf(theta1) - sinf(theta0));
    odom_y  -= r * (cosf(theta1) - cosf(theta0));
  } else {
    float d  = (distL + distR) * 0.5f;
    odom_x  += d * cosf(theta0);
    odom_y  += d * sinf(theta0);
  }
  odom_theta = theta1;

  while (odom_theta >  (float)M_PI) odom_theta -= 2.0f * (float)M_PI;
  while (odom_theta < -(float)M_PI) odom_theta += 2.0f * (float)M_PI;

  // ── Publish odometry ──────────────────────────────────────────────────────
  odom_msg.pose.pose.position.x    = odom_x;
  odom_msg.pose.pose.position.y    = odom_y;
  odom_msg.pose.pose.orientation.z = sinf(odom_theta * 0.5f);
  odom_msg.pose.pose.orientation.w = cosf(odom_theta * 0.5f);
  odom_msg.twist.twist.linear.x    = (velL_filt + velR_filt) * 0.5f;
  odom_msg.twist.twist.angular.z   = dTheta / dt;
  rcl_publish(&pub_odom, &odom_msg, NULL);

  // ── Drift correction ──────────────────────────────────────────────────────
  bool drivingStraight = (fabsf(targetLeft - targetRight) < 0.02f)
                      && (fabsf(targetLeft) > 0.05f);

  if (drivingStraight) {
    float velError = velL_filt - velR_filt;
    driftBias += velError * DRIFT_KI * dt;
    driftBias  = constrain(driftBias, -DRIFT_MAX, DRIFT_MAX);
  } else {
    driftBias *= 0.95f;
  }

  // ── Feedforward PWM + bias ────────────────────────────────────────────────
  int pwmL = vel_to_pwm(targetLeft);
  int pwmR = vel_to_pwm(targetRight);

  if (pwmL == 0 && pwmR == 0) {
    driftBias = 0.0f;
    driveAll(0, 0);
  } else {
    driveAll(constrain(pwmL - (int)driftBias, -PWM_MAX, PWM_MAX),
             constrain(pwmR + (int)driftBias, -PWM_MAX, PWM_MAX));
  }

  // ── Publish wheel speeds (m/s) ────────────────────────────────────────────
  speeds_data[0] = velL_filt;
  speeds_data[1] = velR_filt;
  speeds_msg.data.data = speeds_data;
  speeds_msg.data.size = speeds_msg.data.capacity = 2;
  rcl_publish(&pub_speeds, &speeds_msg, NULL);
}

// ─────────────────────────────────────────────────────────────────────────────
//  CREATE micro-ROS entities
// ─────────────────────────────────────────────────────────────────────────────
bool createEntities() {
  allocator = rcl_get_default_allocator();
  if (rclc_support_init(&support, 0, NULL, &allocator) != RCL_RET_OK) return false;
  if (rclc_node_init_default(&node, "rover_6wd", "", &support) != RCL_RET_OK) return false;

  rclc_subscription_init_default(
      &sub_cmdvel, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist), "/cmd_vel");

  rclc_publisher_init_default(
      &pub_odom, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(nav_msgs, msg, Odometry), "/rover/odom");

  rclc_publisher_init_default(
      &pub_speeds, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
      "/rover/wheel_speeds");

  odom_msg.header.frame_id.data     = ODOM_FRAME;
  odom_msg.header.frame_id.size     = strlen(ODOM_FRAME);
  odom_msg.header.frame_id.capacity = odom_msg.header.frame_id.size + 1;
  odom_msg.child_frame_id.data      = BASE_LINK_FRAME;
  odom_msg.child_frame_id.size      = strlen(BASE_LINK_FRAME);
  odom_msg.child_frame_id.capacity  = odom_msg.child_frame_id.size + 1;

  speeds_msg.data.data     = speeds_data;
  speeds_msg.data.size     = 2;
  speeds_msg.data.capacity = 2;

  rclc_timer_init_default(
      &timer, &support, RCL_MS_TO_NS(CONTROL_DT_MS), timer_callback);

  rclc_executor_init(&executor, &support.context, 2, &allocator);
  rclc_executor_add_subscription(
      &executor, &sub_cmdvel, &cmd_msg, &cmdVelCallback, ON_NEW_DATA);
  rclc_executor_add_timer(&executor, &timer);

  lastCmdTime = millis();
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  DESTROY micro-ROS entities
// ─────────────────────────────────────────────────────────────────────────────
void destroyEntities() {
  stopAll();
  rmw_context_t* rmw_context = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);
  rcl_publisher_fini(&pub_odom,   &node);
  rcl_publisher_fini(&pub_speeds, &node);
  rcl_subscription_fini(&sub_cmdvel, &node);
  rcl_timer_fini(&timer);
  rclc_executor_fini(&executor);
  rcl_node_fini(&node);
  rclc_support_fini(&support);
}

// ─────────────────────────────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  // Disable brownout detector — motor inrush causes brief 3.3V sags.
  // Add 470µF + 100nF caps across the 3.3V rail as hardware protection.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(2000);

  rmw_uros_set_custom_transport(
      true, NULL, serial_open, serial_close, serial_write, serial_read);

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

  // Encoder pins — GPIO34-39 are input-only, no internal pullup
  // External 4.7kΩ pullups to 3.3V required
  pinMode(ENC_L_A, INPUT); pinMode(ENC_L_B, INPUT);
  pinMode(ENC_R_A, INPUT); pinMode(ENC_R_B, INPUT);

  encStateL = (digitalRead(ENC_L_A) << 1) | digitalRead(ENC_L_B);
  encStateR = (digitalRead(ENC_R_A) << 1) | digitalRead(ENC_R_B);

  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrEncL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_L_B), isrEncL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrEncR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_B), isrEncR, CHANGE);

  stopAll();
  delay(1000);
}

// ─────────────────────────────────────────────────────────────────────────────
//  LOOP — reconnection state machine
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  switch (agentState) {

    // ── Waiting for micro-ROS agent on Pi ──────────────────────────────────
    case WAITING_AGENT:
      stopAll();
      if (rmw_uros_ping_agent(PING_TIMEOUT_MS, PING_RETRIES) == RMW_RET_OK) {
        if (createEntities()) {
          agentState = AGENT_CONNECTED;
        } else {
          // Partial allocation must be cleaned up or the next
          // createEntities() call will hit double-init errors.
          destroyEntities();
          delay(500);
        }
      } else {
        delay(500);
      }
      break;

    // ── Normal operation ───────────────────────────────────────────────────
    case AGENT_CONNECTED:
      // Spin executor to handle incoming/outgoing messages.
      // Heartbeats and connection loss are handled internally by the XRCE library.
      if (rclc_executor_spin_some(&executor, RCL_MS_TO_NS(CONTROL_DT_MS)) != RCL_RET_OK) {
        agentState = AGENT_DISCONNECTED;
      }
      break;

    // ── Agent lost — clean up and retry ───────────────────────────────────
    case AGENT_DISCONNECTED:
      stopAll();
      destroyEntities();
      delay(1000);
      agentState = WAITING_AGENT;
      break;
  }
}
