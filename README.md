# IRS_2025_group3

## Table of Contents

- [Pre Build/Run Requirements](#Pre-build/run-requirements)
- [Build](#Build)
- [Run](#Run)

## Pre Build/Run Requirements

### Install Docker

Follow the official instructions [here](https://docs.docker.com/engine/install/)

### Install ROS2 Humble for Ubuntu Jammy Jellyfish (22.04)

Follow the official instructions [here](https://docs.ros.org/en/humble/Installation.html)

Install Eclipse Cyclone DDS (ROS middleware),
more info found [here](https://docs.ros.org/en/humble/Installation/RMW-Implementations/DDS-Implementations/Working-with-Eclipse-CycloneDDS.html)
```bash
    sudo apt install ros-humble-rmw-cyclonedds-cpp
```
Switch from other rmw to rmw_cyclonedds by specifying the environment variable.
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```
## Build

## Run

1. Start irs warehouse simulation, openplc runtime and omron_moma container in one terminal

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
```bash
   http://localhost:8080/
```




