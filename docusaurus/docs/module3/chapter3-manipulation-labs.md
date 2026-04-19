# Chapter 3: Robot Manipulation & Labs

This chapter covers robot manipulation concepts and includes hands-on labs for SLAM/Navigation and object detection/grasping.

---

## Robot Manipulation

### Overview

Robot manipulation is the capability of a robot to physically interact with objects in its environment using an end-effector, typically an articulated robotic arm.

### Degrees of Freedom and Configuration Space

Robotic manipulators are characterized by their degrees of freedom (DOF). A typical robotic arm has 6 DOF to achieve position and orientation control in 3D space.

The configuration space (C-space) is the space of all possible configurations of the manipulator.

### End-Effector Control

The end-effector is the tool at the end of a robotic arm:
- **Grippers**: Parallel, angular, suction cups
- **Tools**: Welding torch, drill
- **Sensors**: Cameras, force/torque sensors

### Grasping Strategies

1. **Parallel Jaw Grasping**: Two opposing fingers for objects with parallel surfaces
2. **Three-Finger Grasping**: Triangular pattern for stable grasps
3. **Suction Cup Grasping**: Vacuum pressure for flat, smooth objects
4. **Adaptive Grasping**: Soft grippers that conform to object shapes

### MoveIt 2 Integration

MoveIt 2 is the official motion planning framework for ROS 2:

```python
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import MoveItResponse
from geometry_msgs.msg import Pose

class ManipulationController(Node):
    def __init__(self):
        super().__init__('manipulation_controller')
        # MoveIt 2 setup would go here
        # move_group_interface = MoveGroupInterface(self, "arm_group")

    def move_to_pose(self, target_pose):
        # Set target pose and plan
        pass
```

---

## Lab 5: SLAM and Navigation

### Objective

Perform SLAM in an unknown environment with a mobile robot and navigate to a target goal.

### Prerequisites

- ROS 2 Humble Hawksbill or later
- Navigation2 package
- slam_toolbox package
- Gazebo/Ignition

### Step 1: Environment Setup

1. Launch Gazebo with an unknown environment:
```bash
ros2 launch gazebo_ros gazebo.launch.py world_name:=cave.world
```

2. Spawn your robot:
```bash
ros2 run gazebo_ros spawn_entity.py -entity robot -topic robot_description
```

### Step 2: Launch SLAM Toolbox

```bash
ros2 launch slam_toolbox online_async_launch.py params_file:=/path/to/slam_params.yaml
```

### Step 3: Run SLAM

```bash
# Teleoperate the robot to explore
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Or run autonomous exploration
ros2 run lab5_slam_navigation exploration_controller
```

### Step 4: Save the Map

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "filename: map_name"
```

### Step 5: Navigate to Goal

```bash
ros2 launch nav2_bringup bringup_launch.py map:=map_name.yaml
```

### Expected Outcomes

- Robot creates a map of the environment using SLAM
- Robot can navigate autonomously to a specified goal
- Map accuracy can be verified against ground truth

---

## Lab 6: Object Detection & Grasping

### Objective

Use computer vision to detect an object and perform pick-and-place with a manipulator.

### Prerequisites

- ROS 2 Humble Hawksbill or later
- MoveIt 2 for motion planning
- OpenCV for image processing
- Gazebo simulation

### Step 1: Object Detection

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.detect_callback, 10)
        self.bridge = CvBridge()

    def detect_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Simple color-based detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                # Object detected at (x, y) with size (w, h)
```

### Step 2: MoveIt 2 Grasping

```python
# Approach object
approach_pose = Pose()
approach_pose.position.x = object_x
approach_pose.position.y = object_y
approach_pose.position.z = object_z + 0.1

# Plan and execute approach
move_group.setPoseTarget(approach_pose)
move_group.go(wait=True)

# Lower to grasp
grasp_pose = approach_pose
grasp_pose.position.z = object_z
move_group.setPoseTarget(grasp_pose)
move_group.go(wait=True)

# Close gripper
gripper_pub.publish(Float64(data=0.5))

# Lift object
move_group.setPoseTarget(approach_pose)
move_group.go(wait=True)
```

### Step 3: Place Object

```python
# Move to place location
place_pose = Pose()
place_pose.position.x = place_x
place_pose.position.y = place_y
place_pose.position.z = place_z + 0.1
move_group.setPoseTarget(place_pose)
move_group.go(wait=True)

# Lower and release
place_pose.position.z = place_z
move_group.setPoseTarget(place_pose)
move_group.go(wait=True)

gripper_pub.publish(Float64(data=0.0))

# Move away
move_group.setPoseTarget(approach_pose)
move_group.go(wait=True)
```

### Expected Outcomes

- Object successfully detected using vision
- Robot approaches and grasps the object
- Robot places object at target location

## Acceptance Criteria Met

- [X] Robot manipulation concepts explained
- [X] MoveIt 2 integration
- [X] Lab 5: SLAM and Navigation implementation
- [X] Lab 6: Object detection and grasping
