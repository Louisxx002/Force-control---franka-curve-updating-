#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "${1:-}" == "check_approach" || "${1:-}" == "planning" || "${1:-}" == "planning_probe" || "${1:-}" == "load_scene" || "${1:-}" == "depth_scene" ]]; then
    # ROS Python packages belong to the system interpreter, not the demo venv.
    set +u
    source /opt/ros/jazzy/setup.bash
    source /home/pnp/franka_ros2_ws/install/setup.bash
    set -u
    if [[ "$1" == "planning" ]]; then
        exec ros2 launch ./launch/planning.launch.py "${@:2}"
    fi
    exec /usr/bin/python3 -m "curve_wipe.$1" "${@:2}"
fi
exec .venv/bin/python -m "curve_wipe.${1:?Use preview, capture, plan, autowipe, sensor, execute or adaptive_execute}" "${@:2}"
