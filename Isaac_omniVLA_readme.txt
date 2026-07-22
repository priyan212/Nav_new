Launch isaac sim:


#!/bin/bash
# 1. Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate isaacsim


# 2. Force EULA acceptance (Environment Variable)
export OMNI_KIT_ACCEPT_Eros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.8}, angular: {z: 0.5}}"EULA=YES


# 3. Fix paths for ROS 2 Jazzy (Internal 3.11 libs)
export ISAAC_SIM_ROOT=/mnt/bigdisk/conda_envs/isaacsim/lib/python3.11/site-packages/isaacsim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/jazzy/lib
export PYTHONPATH=$PYTHONPATH:$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/jazzy/lib/python3.11/site-packages


# 4. Run Isaac Sim (Added --accept-eula flag as a backup)
# Also added --no-window if you want to save RAM (Headless)
isaacsim --accept-eula "$@"




Open the recent usd – 6wheeled_rover.usd
(wait for it to load)


Press play

Open the script editor and run the two below 
exec(open("/mnt/bigdisk/motion_planning/scripts/isaac_cmdvel_bridge.py").read(), globals())
exec(open("/mnt/bigdisk/motion_planning/scripts/isaac_camera_ros2_pub.py").read(), globals())





Outside open new terminal and follow the below


Terminal 1 (ROS 2):

source /opt/ros/jazzy/setup.bash
cd /mnt/bigdisk/motion_planning/OmniVLA
./zenoh-bridge-ros2dds -c configs/zenoh_isaac_bridge.json5

Terminal 2 (omnivla env):
  conda activate omnivla
  LD_LIBRARY_PATH=/mnt/bigdisk/conda_envs/omnivla/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
    python inference/isaac_gui.py
