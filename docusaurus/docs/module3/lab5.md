---
sidebar_position: 8
---

# Lab 5: SLAM and Navigation

## Objective
Perform SLAM in an unknown environment with a mobile robot and navigate to a target goal.

## Learning Outcomes
By the end of this lab, students will be able to:
- Set up and configure the ROS 2 Navigation Stack for SLAM
- Execute SLAM in a Gazebo simulation environment
- Plan and execute autonomous navigation to a specified goal
- Evaluate the accuracy of the generated map

## Prerequisites
- Module 1: ROS 2 Fundamentals
- Module 2: Robotics Simulation
- Module 3: SLAM and Navigation Stack

## Hardware Requirements
- Mobile robot platform with a 2D lidar (e.g., TurtleBot 4 or similar)
- Computer with ROS 2 and Gazebo installed
- (Alternative) Simulated mobile robot in Gazebo

## Software Dependencies
- ROS 2 Humble Hawksbill or later
- Navigation2 package
- slam_toolbox package
- Gazebo/Ignition
- RViz2

## Background Theory

### Simultaneous Localization and Mapping (SLAM)
SLAM is the computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it. In robotics, this typically involves using sensor data to build a map of the environment while determining the robot's position within that map.

### Navigation Stack
The ROS 2 Navigation Stack provides a set of packages that implement navigation functionality for mobile robots, including:
- Localization (AMCL)
- Global and local path planning
- Obstacle avoidance
- Path execution

## Lab Procedure

### Step 1: Environment Setup
1. Launch a Gazebo simulation with an unknown environment:
   ```bash
   ros2 launch turtlebot4_ignition_bringup turtlebot4_world.launch.py world:=maze
   ```

2. Launch the robot's sensor configuration:
   ```bash
   ros2 launch turtlebot4_ignition_bringup turtlebot4_sensor_fusion.launch.py
   ```

### Step 2: SLAM Configuration
1. Create a SLAM launch file (`slam.launch.py`) with appropriate parameters:
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')
       slam_params_file = LaunchConfiguration('slam_params_file')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation/Gazebo clock'),
           DeclareLaunchArgument(
               'slam_params_file',
               default_value='/path/to/slam_params.yaml',
               description='Full path to the slamtoolbox parameter file to use'),

           Node(
               package='slam_toolbox',
               executable='async_slam_toolbox_node',
               parameters=[slam_params_file, {'use_sim_time': use_sim_time}],
               output='screen')
       ])
   ```

2. Create a parameter file (`slam_params.yaml`) for SLAM configuration:
   ```yaml
   slam_toolbox:
     ros__parameters:
       use_sim_time: true
       # Plugin params
       solver_plugin: solver_plugins::CeresSolver
       ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
       # ROS params
       odom_frame: odom
       map_frame: map
       base_frame: base_link
       scan_topic: /scan
       # Other params...
   ```

### Step 3: Execute SLAM
1. Launch the SLAM node:
   ```bash
   ros2 launch your_package slam.launch.py
   ```

2. Launch RViz2 for visualization:
   ```bash
   ros2 run rviz2 rviz2 -d /path/to/slam_config.rviz
   ```

3. Navigate the robot to explore the environment:
   ```bash
   ros2 run teleop_twist_keyboard teleop_twist_keyboard
   ```

4. Monitor the map building process in RViz2.

### Step 4: Navigation Setup
1. Once the map is sufficiently built, configure the Navigation Stack:
   ```bash
   ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true
   ```

2. Launch the navigation bringup with localization:
   ```bash
   ros2 launch nav2_bringup localization_launch.py use_sim_time:=true map:=/path/to/your/map.yaml
   ```

### Step 5: Autonomous Navigation
1. Use RViz2 to set a 2D Pose Estimate for initial localization.

2. Use the "Nav2 Goal" tool in RViz2 to set a navigation goal.

3. Observe the robot's path planning and execution.

4. Monitor the robot's performance metrics (path efficiency, obstacle avoidance, etc.).

## Expected Results
- A coherent map of the unknown environment generated through SLAM
- Successful autonomous navigation to the specified goal
- Robot's ability to avoid obstacles while following the planned path
- Accurate localization within the generated map

## Troubleshooting Tips
- If the map appears distorted, check the robot's odometry accuracy
- If navigation fails, verify that the costmaps are properly configured
- If the robot gets stuck, adjust the local planner parameters (DWA, TEB, etc.)
- Ensure proper TF tree connections between all coordinate frames

## Assessment Questions
1. How did the quality of the SLAM-generated map change as you explored more of the environment?
2. What factors influenced the robot's ability to successfully navigate to the goal?
3. How did the robot handle dynamic obstacles in the environment?
4. What improvements could be made to the SLAM parameters to enhance map quality?

## Extensions
- Implement dynamic obstacle avoidance during navigation
- Compare different SLAM algorithms (Gmapping, Cartographer, SLAM Toolbox)
- Add semantic information to the generated map (object detection and labeling)
- Implement multi-session mapping with map merging

## References
- ROS 2 Navigation Documentation
- SLAM Toolbox Documentation
- Navigation2 Tutorials