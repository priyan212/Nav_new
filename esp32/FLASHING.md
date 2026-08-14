# Flashing `rover_6wd_complete.ino` onto the rover's ESP32

The ESP32 is wired to the **Raspberry Pi's** `/dev/ttyUSB0`, not to the GPU
machine, so flashing is a compile-locally → copy-to-Pi → flash-via-SSH
workflow, not a direct USB upload. Confirmed working end-to-end 2026-08-06
(BNO055 calibration-persistence firmware update) and again 2026-08-14
(BNO055 → BNO085 IMU rewrite, see step 1's library note below).

## 1. Compile locally on the GPU machine

Classic Arduino IDE 1.8.19 is installed at `/usr/bin/arduino` (no
`arduino-cli` on this machine or the Pi — don't be misled by a leftover
`~/.arduino15/arduino-cli.yaml` config file into thinking arduino-cli itself
is installed, it isn't). It compiles headlessly fine as long as an X display
is available (`echo $DISPLAY`) — no Xvfb needed on this machine.

**Gotcha:** Arduino's build system merges *every* `.ino` file in the
sketch's containing folder into one translation unit. This `esp32/` folder
has multiple standalone sketches side by side (`rover_6wd_complete.ino`,
`rover_6wd_microros_handshake.ino`, etc.) — compiling directly from this
folder fails with duplicate-symbol errors (`redefinition of 'void setup()'`
etc.). Fix: copy just the target `.ino` into an isolated folder named
identically to the sketch (Arduino requires folder name == main `.ino`
basename) before compiling:

```bash
mkdir -p /tmp/esp32_build/rover_6wd_complete
cp esp32/rover_6wd_complete.ino /tmp/esp32_build/rover_6wd_complete/
mkdir -p /tmp/esp32_build_out
/usr/bin/arduino --verify /tmp/esp32_build/rover_6wd_complete/rover_6wd_complete.ino \
  --board esp32:esp32:esp32 --pref build.path=/tmp/esp32_build_out
```

This produces `rover_6wd_complete.ino.bin` / `.bootloader.bin` /
`.partitions.bin` in `build.path`. `boot_app0.bin` is a fixed OTA-selector
stub that doesn't change between builds — reuse the one already on the Pi
(see step 2) instead of hunting for it locally.

**Library dependency (added 2026-08-14, BNO085 rewrite):** the sketch now
`#include <Adafruit_BNO08x.h>`, which is NOT bundled with the ESP32 core or
`micro_ros_arduino` — unlike everything else this sketch uses. If
`~/Arduino/libraries/` on the GPU machine doesn't already have
`Adafruit_BNO08x`, `Adafruit_BusIO`, and `Adafruit_Sensor`, the compile
fails with a missing-header error. Install once with:

```bash
cd ~/Arduino/libraries
git clone --depth 1 https://github.com/adafruit/Adafruit_BNO08x.git
git clone --depth 1 https://github.com/adafruit/Adafruit_BusIO.git
git clone --depth 1 https://github.com/adafruit/Adafruit_Sensor.git
```

No version pin exists for these yet (unlike `micro_ros_arduino`'s pinned
install) — if a future recompile behaves differently, check what commit is
actually checked out in each of these three directories first.

## 2. Copy the 3 new artifacts to the Pi

The Pi already has prior build dirs at `~/esp32_flash/{original_build,
calib_build,imu_build}/` (each with `.bin`/`.bootloader.bin`/
`.partitions.bin`) plus older `~/fw_flash/` and `~/rover_fw/` dirs (each
with a `boot_app0.bin` + working `flash.sh`/`do_flash.sh` scripts using the
same esptool invocation as step 4). Make a new subdir under
`~/esp32_flash/` for this build, `scp` the 3 new files in, then `cp`
`boot_app0.bin` from any prior build dir (e.g.
`~/esp32_flash/imu_build/boot_app0.bin`) into the new one.

## 3. Stop the systemd agent service — not just `pkill`

`rover-agent` (systemd, `Restart=always`) respawns `micro_ros_agent` within
~1s of a plain `pkill`, silently re-grabbing `/dev/ttyUSB0` and making
esptool hang/fail against a port it can never get exclusive access to. This
produces a command that looks "stuck" with no error. Always:

```bash
sudo systemctl stop rover-agent   # NOT just pkill -f micro_ros_agent
```

then verify with `systemctl is-active rover-agent` (expect `inactive`)
before flashing.

## 4. Flash

esptool.py v2.8 is already installed on the Pi (`/usr/bin/esptool`):

```bash
sudo esptool --chip esp32 --port /dev/ttyUSB0 --baud 115200 \
  --before default_reset --after hard_reset write_flash -z \
  --flash_mode dio --flash_freq 40m --flash_size detect \
  0x1000  bootloader.bin \
  0x8000  partitions.bin \
  0xe000  boot_app0.bin \
  0x10000 <sketch>.ino.bin
```

Success signature: `Hash of data verified.` after each of the 4 segments,
ending `Leaving... / Hard resetting via RTS pin...`, exit 0. Takes ~25-35s
total (the app partition dominates).

## 5. Verify

```bash
sudo systemctl restart rover-agent
# then, after a few seconds:
ros2 topic echo /rover/rpm --once
```

Should show live `[left_rpm, right_rpm, imu_heading_deg, imu_calib]` data.

## Resetting the ESP32 without reflashing

To force a clean reboot without writing new firmware, stop the agent and
run esptool with a no-op command — `chip_id` just connects (which resets
into the bootloader) and then hard-resets back into the app via
`--after hard_reset`:

```bash
sudo systemctl stop rover-agent
sudo esptool --chip esp32 --port /dev/ttyUSB0 \
  --before default_reset --after hard_reset chip_id
sudo systemctl restart rover-agent
```

Confirmed working 2026-08-06: used this to verify the (then-current) BNO055
calibration offsets survived a reset — `acc` calibration jumped straight to
3 on reboot instead of requiring the full pose dance again. As of the
2026-08-14 BNO085 rewrite, calibration persistence works differently (the
sensor's own SH-2 firmware owns it in its own flash via
`sh2_setDcdAutoSave(true)`, not ESP32-side NVS) — this reset trick still
works to force a clean reboot, just via a different underlying mechanism.

## Known flakiness

- **SSH commands intermittently die with exit 255** for no real reason —
  including RIGHT AFTER a command has already succeeded remotely (seen
  during the 2026-08-06 flash: `systemctl stop rover-agent` reported exit
  255 but had actually already stopped the service). Always re-check actual
  remote state (`pgrep -af micro_ros_agent`, `systemctl is-active
  rover-agent`) after a 255 before assuming the action failed and retrying
  it.
- **Password prompts**: use `sudo -S -p ""` (empty prompt) when piping the
  password over SSH — e.g. `echo "$PI_PASS" | sudo -S -p "" cmd`. Without
  `-p ""`, the literal `[sudo] password for pi:` prompt text sometimes
  leaks into stderr in a way that reads as a failure even when a bare
  `sudo -S` alone works fine.
