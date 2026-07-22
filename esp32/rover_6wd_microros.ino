/*
 * rover_6wd_microros.ino
 * 
 * Micro-ROS firmware for 6-wheel rover with encoders
 * Publishes /rover/rpm as Float32MultiArray [left_rpm, right_rpm]
 * 
 * ESP32 pin connections:
 *   - Left encoder:  GPIO 35 (encoder phase A)
 *   - Right encoder: GPIO 34 (encoder phase A)  
 *   - Serial TX: GPIO 1 (USB serial for ROS2)
 *   - Serial RX: GPIO 3 (USB serial for ROS2)
 */

#include <Arduino.h>
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32_multi_array.h>

// Encoder pins
#define LEFT_ENCODER_PIN 35
#define RIGHT_ENCODER_PIN 34

// Encoder state
volatile long left_pulse_count = 0;
volatile long right_pulse_count = 0;
unsigned long last_time = 0;

// ROS2 objects
rcl_publisher_t publisher;
std_msgs__msg__Float32MultiArray rpm_msg;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

// ============================================================================
// Encoder ISR callbacks
// ============================================================================

void IRAM_ATTR left_encoder_isr() {
  left_pulse_count++;
}

void IRAM_ATTR right_encoder_isr() {
  right_pulse_count++;
}

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n=== rover_6wd micro-ROS starting ===");
  
  // Setup encoder pins
  pinMode(LEFT_ENCODER_PIN, INPUT);
  pinMode(RIGHT_ENCODER_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_PIN), left_encoder_isr, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_PIN), right_encoder_isr, RISING);
  
  Serial.println("[OK] Encoder pins initialized");
  
  // Initialize micro-ROS
  set_microros_transports();
  delay(2000);
  
  allocator = rcl_get_default_allocator();
  
  // Create init options
  rclc_support_init(&support, 0, NULL, &allocator);
  
  // Create node
  rclc_node_init_default(&node, "rover_6wd", "", &support);
  
  Serial.println("[OK] ROS2 node initialized: /rover_6wd");
  
  // Create publisher for /rover/rpm
  rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "/rover/rpm"
  );
  
  Serial.println("[OK] Publisher /rover/rpm created");
  
  // Initialize message with 2 elements
  rpm_msg.data.size = 0;
  rpm_msg.data.capacity = 2;
  rpm_msg.data.data = (float *)malloc(rpm_msg.data.capacity * sizeof(float));
  
  // Pre-allocate data
  if (rpm_msg.data.data == NULL) {
    Serial.println("[ERROR] Failed to allocate memory for rpm_msg");
    while (1) delay(100);
  }
  
  // Add two float slots
  float *ptr = rpm_msg.data.data;
  ptr[0] = 0.0f;  // left RPM
  ptr[1] = 0.0f;  // right RPM
  rpm_msg.data.size = 2;
  
  Serial.println("[OK] Message memory allocated");
  Serial.println("\n=== rover_6wd ready - publishing /rover/rpm ===\n");
  
  last_time = millis();
}

// ============================================================================
// Loop
// ============================================================================

void loop() {
  unsigned long current_time = millis();
  unsigned long dt = current_time - last_time;
  
  // Publish at 10 Hz (every 100 ms)
  if (dt >= 100) {
    // Calculate RPM
    // Assume: 20 counts per revolution
    // RPM = (counts / 20) / (time_in_seconds) * 60
    
    float left_rpm = (float)left_pulse_count / 20.0f / (dt / 1000.0f) * 60.0f;
    float right_rpm = (float)right_pulse_count / 20.0f / (dt / 1000.0f) * 60.0f;
    
    // Reset counters
    left_pulse_count = 0;
    right_pulse_count = 0;
    last_time = current_time;
    
    // Fill message
    rpm_msg.data.data[0] = left_rpm;
    rpm_msg.data.data[1] = right_rpm;
    
    // Publish
    rcl_ret_t ret = rcl_publish(&publisher, &rpm_msg, NULL);
    
    if (ret != RCL_RET_OK) {
      Serial.print("[WARN] Publish failed: ");
      Serial.println(ret);
    } else {
      Serial.print("[PUB] L:");
      Serial.print(left_rpm, 1);
      Serial.print(" R:");
      Serial.print(right_rpm, 1);
      Serial.println(" RPM");
    }
  }
  
  delay(10);
}

/*
 * NOTES FOR COMPILATION:
 * 
 * 1. Install Arduino IDE with ESP32 support:
 *    - Open Arduino IDE
 *    - File → Preferences
 *    - Additional Board URLs: 
 *      https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *    - Tools → Board Manager → Search "esp32" → Install
 * 
 * 2. Install micro-ROS Arduino library:
 *    - Sketch → Include Library → Manage Libraries
 *    - Search "micro_ros_arduino" → Install
 *    - Choose micro-ROS agent for your transport (serial)
 * 
 * 3. Select board and port:
 *    - Tools → Board → "ESP32 Dev Module"
 *    - Tools → Port → /dev/ttyUSB0 (or your USB port)
 *    - Tools → Upload Speed → 115200
 * 
 * 4. Upload:
 *    - Sketch → Upload (Ctrl+U)
 *    - Wait 1-2 minutes for compilation and upload
 * 
 * 5. Verify:
 *    - Tools → Serial Monitor → 115200 baud
 *    - Should see: "=== rover_6wd micro-ROS starting ===" messages
 *    - Should see: "[PUB] L:0.0 R:0.0 RPM" messages
 */
