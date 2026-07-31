# Priyan's Notes — Quick Launch Cheatsheet

> Personal scratch notes, not the full docs — see [README.md](README.md) for that.

##  GUI only (Isaac Sim connects to this)

Just to see the Navigation running. Isaac Sim connects to this GUI —
see `Isaac_omniVLA_readme.txt`.

```bash
cd /mnt/bigdisk/Priyan/Nav_new
./launch_gui.sh
```

##  Actual launch file (real rover)

```bash
cd /mnt/bigdisk/Priyan/Nav_new
./launch_rover.sh
```

##  MARS (Habitat sim)

```bash
cd /mnt/bigdisk/Priyan/Nav_new/MARS
./launch_mars.sh
./launch_mars.sh --belief-only --rocks
```

# Earth habitat sim (real-world photogrammetry scan, same conda env as Mars)

```
cd /mnt/bigdisk/Priyan/Nav_new
./EARTH/launch_earth.sh

```
# Launch script for the repeat is isaac sim
./ISAAC/launch_isaac_topo_repeat.sh --arrival-sim 0.85

# Supervised live-rover A/B test (once 1-3 look right, with you present):

./launch_rover.sh                        # belief ON (new default)
./launch_rover.sh --no-belief-goal        # old frozen-goal behavior, for comparison

# newlaunch file: launch_rover_vitb.sh — thin wrapper around launch_rover.sh (same Pi bringup/health checks) that just adds --depth-encoder vitb:
./launch_rover_vitb.sh                          # default Pi IP
./launch_rover_vitb.sh 10.47.234.125 --target "trash bin"