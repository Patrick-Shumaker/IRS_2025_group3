# IRS_2025_group3

## Table of Contents

- [Pre Build/Run Requirements](#Pre-build/run-requirements)
- [Build](#Build)
- [Run](#Run)

## Pre Build/Run Requirements

## Build

## Run

1. Start irs warehouse simulation, openplc runtime and omron container in one terminal

```bash
cd ~/industrial-robots-and-systems-world
xhost +local:root
docker compose up
```

2. Start launch file in 2nd terminal, click 2D pose estimate in RViz and select origin (may not need to)
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch hand_solo_virtual_nav nav_launch.py
```

3. Open browser and enter the following into url:
4. ```bash
   http://localhost:8080/
   ```
