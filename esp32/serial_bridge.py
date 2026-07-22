#!/usr/bin/env python3
"""Simple serial-to-TCP bridge. Run on Pi to expose /dev/ttyUSB0 over TCP."""
import socket, serial, threading, sys, time

PORT = 4444
SERIAL_DEV = "/dev/ttyUSB0"
BAUD = 115200

def pipe(src, dst, label):
    try:
        while True:
            data = src.read(4096) if hasattr(src, 'read') else src.recv(4096)
            if not data:
                break
            if hasattr(dst, 'write'):
                dst.write(data)
            else:
                dst.sendall(data)
    except Exception as e:
        print(f"[{label}] pipe ended: {e}")

print(f"Opening {SERIAL_DEV} at {BAUD} baud...")
ser = serial.Serial(SERIAL_DEV, BAUD, timeout=0.1)
print(f"Listening on TCP port {PORT}...")

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT))
srv.listen(1)
print("Waiting for connection...")

conn, addr = srv.accept()
print(f"Connected from {addr}")
conn.setblocking(True)

t1 = threading.Thread(target=pipe, args=(ser, conn, "serial->tcp"), daemon=True)
t2 = threading.Thread(target=pipe, args=(conn, ser, "tcp->serial"), daemon=True)
t1.start(); t2.start()
t1.join(); t2.join()
print("Done.")
