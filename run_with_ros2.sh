#!/bin/bash
# Run franka_ycb_pick_place.py with ROS2 Bridge support
# This script sets up the environment for Isaac Sim's internal ROS2 libraries

# Isaac Sim path
ISAAC_SIM_PATH=~/isaac_sim_4.5

# ROS2 Bridge environment variables (required for Isaac Sim's internal ROS2)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISTRO=humble
export LD_LIBRARY_PATH="$ISAAC_SIM_PATH/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH"

echo "[ROS2 Setup] Environment configured:"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  ROS_DISTRO=$ROS_DISTRO"
echo "  LD_LIBRARY_PATH includes Isaac Sim ROS2 libraries"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run with or without --ros2 flag based on arguments
cd "$ISAAC_SIM_PATH"
./python.sh "$SCRIPT_DIR/franka_ycb_pick_place.py" --ros2 "$@"
