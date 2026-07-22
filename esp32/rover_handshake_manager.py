#!/usr/bin/env python3
"""
Rover ESP32 Handshake Manager
Runs on RPi to detect and establish connection with ESP32
"""

import serial
import time
import sys
from threading import Thread, Event

class RoverHandshakeManager:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = Event()
        self.running = Event()
        self.running.set()
        
        # Handshake constants
        self.MAGIC = 0xABCD
        self.HEARTBEAT_TIMEOUT = 3000  # 3 seconds
        self.last_heartbeat = time.time()
        
    def connect(self):
        """Establish serial connection to ESP32"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"✓ Serial connection opened on {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"✗ Failed to open serial: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ Serial connection closed")
            self.connected.clear()
    
    def send_ack(self):
        """Send acknowledgment to ESP32"""
        try:
            ack = bytes([0xAA, 0xBB])  # ACK marker
            self.ser.write(ack)
            self.last_heartbeat = time.time()
            return True
        except Exception as e:
            print(f"✗ Failed to send ACK: {e}")
            return False
    
    def parse_beacon(self, data):
        """
        Parse handshake beacon from ESP32
        Format: [MAGIC_HIGH, MAGIC_LOW, VERSION, ROVER_ID]
        """
        if len(data) >= 4:
            magic = (data[0] << 8) | data[1]
            version = data[2]
            rover_id = data[3]
            
            if magic == self.MAGIC:
                print(f"✓ Valid beacon received: Version={version}, RoverID={rover_id}")
                return True
        return False
    
    def listen_thread(self):
        """Background thread to listen for ESP32 messages"""
        print("→ Starting listener thread...")
        buffer = bytearray()
        
        while self.running.is_set():
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting)
                    buffer.extend(data)
                    
                    # Check for text messages (debug info)
                    if b'\n' in buffer:
                        lines = buffer.split(b'\n')
                        for line in lines[:-1]:
                            msg = line.decode('utf-8', errors='ignore').strip()
                            if msg:
                                print(f"  [ESP32] {msg}")
                        buffer = bytearray(lines[-1])
                    
                    # Check for beacon (4 bytes)
                    if len(buffer) >= 4:
                        # Look for beacon pattern
                        if buffer[0:2] == bytes([0xAB, 0xCD]):
                            if self.parse_beacon(buffer[0:4]):
                                self.connected.set()
                                self.send_ack()
                                buffer = buffer[4:]
                        # Check for heartbeat
                        elif buffer[0:2] == bytes([0xFF, 0xFE]):
                            self.last_heartbeat = time.time()
                            print("  ♥ Heartbeat received")
                            buffer = buffer[2:]
                        else:
                            # Skip invalid byte
                            buffer = buffer[1:]
                
                # Check heartbeat timeout
                if self.connected.is_set():
                    if (time.time() - self.last_heartbeat) > (self.HEARTBEAT_TIMEOUT / 1000.0):
                        print("✗ Heartbeat timeout - connection lost")
                        self.connected.clear()
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"✗ Listener error: {e}")
                self.connected.clear()
                time.sleep(0.5)
    
    def start(self):
        """Start handshake process"""
        print("\n" + "="*70)
        print("  ROVER ESP32 HANDSHAKE MANAGER")
        print("="*70)
        
        if not self.connect():
            return False
        
        # Start listener thread
        listener = Thread(target=self.listen_thread, daemon=True)
        listener.start()
        
        # Wait for connection
        print(f"→ Waiting for ESP32 beacon ({self.timeout}s timeout)...")
        
        if self.connected.wait(timeout=self.timeout):
            print("✓ HANDSHAKE COMPLETE - ESP32 ready")
            return True
        else:
            print("✗ HANDSHAKE FAILED - No beacon received")
            print("  Troubleshooting:")
            print("  1. Check USB cable connection")
            print("  2. Verify /dev/ttyUSB0 is accessible")
            print("  3. Check ESP32 has 115200 baud serial configured")
            print("  4. Press RESET button on ESP32")
            return False
    
    def monitor(self, duration=30):
        """Monitor connection health for specified duration"""
        print(f"\n→ Monitoring connection for {duration} seconds...")
        print("  (Ctrl+C to stop)\n")
        
        start_time = time.time()
        status_shown = False
        
        try:
            while (time.time() - start_time) < duration:
                elapsed = time.time() - start_time
                
                if self.connected.is_set():
                    status = "✓ CONNECTED"
                    heartbeat_age = (time.time() - self.last_heartbeat) * 1000
                    print(f"[{elapsed:6.1f}s] {status} (heartbeat: {heartbeat_age:6.0f}ms ago)")
                else:
                    print(f"[{elapsed:6.1f}s] ✗ DISCONNECTED")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n[Monitoring stopped]")
    
    def stop(self):
        """Stop all operations"""
        self.running.clear()
        time.sleep(0.1)
        self.disconnect()


def main():
    # Auto-detect port if not specified
    port = '/dev/ttyUSB0'
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    manager = RoverHandshakeManager(port=port, timeout=5)
    
    try:
        if manager.start():
            # Monitor for 30 seconds
            manager.monitor(duration=30)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
    finally:
        manager.stop()


if __name__ == '__main__':
    main()
