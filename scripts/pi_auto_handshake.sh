#!/bin/bash
# Runs ON THE PI. Establishes the ESP32 micro-ROS session automatically:
#   1. kills any duplicate micro-ROS agents (they fight over the serial port)
#   2. hardware-resets the ESP32 by pulsing RTS (same auto-reset circuit the
#      flash tool uses — replaces pressing the physical RESET button)
#   3. starts exactly ONE agent, detached (survives SSH drops)
#   4. waits until the agent reports a client session
# Writes progress to /tmp/handshake.log; prints HANDSHAKE_OK / HANDSHAKE_TIMEOUT.
exec > /tmp/handshake.log 2>&1

PORT=${1:-/dev/ttyUSB0}

pkill -9 -f micro_ros_agent 2>/dev/null
sleep 2

python3 - "$PORT" << 'EOF'
import serial, sys, time
port = sys.argv[1]
s = serial.Serial(port, 115200)
s.dtr = False
s.rts = True     # EN low -> hold chip in reset
time.sleep(0.2)
s.rts = False    # EN high -> boot
time.sleep(0.5)
s.close()
print("ESP32 reset pulse sent on", port)
EOF

source /opt/ros/humble/setup.bash
source ~/uros_ws/install/setup.bash 2>/dev/null || source ~/microros_ws/install/setup.bash
nohup ros2 run micro_ros_agent micro_ros_agent serial --dev "$PORT" -b 115200 -v4 \
    > /tmp/agent.log 2>&1 < /dev/null &
disown

for i in $(seq 1 25); do
    sleep 1
    if grep -qiE "session established|create.*session|client_key" /tmp/agent.log; then
        echo HANDSHAKE_OK
        exit 0
    fi
done
echo HANDSHAKE_TIMEOUT
tail -5 /tmp/agent.log
exit 1
