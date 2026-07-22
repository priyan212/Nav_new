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
./launch_mars.sh
./launch_mars.sh --belief-only --rocks
```
