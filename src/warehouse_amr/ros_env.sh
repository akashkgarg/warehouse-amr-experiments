#!/usr/bin/env bash
# Run a command in the ROS 2 Jazzy environment used by nav2_bridge.py, on an isolated localhost-only domain.
#   tools/warehouse_amr/ros_env.sh python3.12 tools/warehouse_amr/nav2_bridge.py nav2 ...
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-57}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RCUTILS_COLORIZED_OUTPUT=0
exec "$@"
