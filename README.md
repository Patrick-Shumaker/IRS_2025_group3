# IRS_2025_group3

State machine for autonmous operation of an omron mobile robot to pick boxes off conveyor  
at specific positions and place on shelf while avoiding dynamic/static obstacles continuously.

Built for the simulation environment by [Collabrotiive Robotics Lab](https://github.com/CollaborativeRoboticsLab)

## Table of Contents

- [Pre Build/Run Requirements](#Pre-build/run-requirements)
- [Build](#Build)
- [Run](#Run)

## Pre Build/Run Requirements

### Install Docker

Follow the official instructions [here](https://docs.docker.com/engine/install/)

State machine built for the simulation environment by Collaborative Robotics Lab found [here](https://github.com/CollaborativeRoboticsLab/industrial-robots-and-systems-world)  

In the terminal run the following command to clone the repo

```bash
git clone https://github.com/CollaborativeRoboticsLab/industrial-robots-and-systems-world.git
```

Enter the folder
```bash
cd industrial-robots-and-systems-world
```

Pull the latest docker containers
```bash
docker compose pull
```
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
Install Moveit2 for ROS2 Humble
```bash
sudo apt install ros-humble-moveit
```

Install Nav2 for ROS2 Humble
```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

## Build

In home directory (or preferred) make a ros workspace e.g.
```bash
cd ~/
mkdir irs-workspace
```

In the workspace create a source file
```bash
cd ~/irs-workspace
mkdir src
```

In the source file clone the github repo
```bash
cd ~/irs-workspace/src
git clone https://github.com/Patrick-Shumaker/IRS_2025_group3.git
```

Navigate to the workspace and build ROS packages
```bash
cd ~/irs-workspace
colcon build
```

## Run

1. Start irs warehouse simulation, openplc runtime and omron_moma container in one terminal

```bash
cd ~/industrial-robots-and-systems-world
xhost +local:root
docker compose up
```
Ensure arm successfuly initialised in RViz, if grey links are present compose down and retry  
until succesful initialisation. (Will cause arm movement errors otherwise)

2. Start launch file in 2nd terminal, click 2D pose estimate in RViz and select origin (may not need to)
```bash
cd ~/irs-workspace/
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch hand_solo_virtual_nav nav_launch.py
```

3. Open browser and open [http://localhost:8080/](http://localhost:8080/login)
Username/password is 'openplc'

Click 'Programs' in sidebar and click 'Browse' under upload programs header,  
upload 'timerbelt.st' (located in IRS_2025_group3 directory).  

Navigate to dashboard in the sidebar and click 'Start PLC'

4. In the irs simulation press 'P' to open the HMI, press 'Start PLC' to enable interface buttons/conveyor (Green indicators show success).
Spawn a box, depending on the box size it should stop at one of three locations (A=big, B=med, C=small)

5. Press 'R' to enable autonomous mode and add Kevin 'dynamic obstacle'

6. In a third terminal run the state machine node to start operation
```bash
cd ~/irs-workspace/
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 run hand_solo_virtual_nav hs_waypoint_follower
```




