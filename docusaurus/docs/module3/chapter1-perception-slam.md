# Chapter 1: Perception & SLAM

Welcome to Module 3 of the Physical AI & Humanoid Robotics Textbook. This module focuses on the artificial intelligence aspects of robotics, covering perception systems, navigation, and manipulation.

## Overview

In this module, you will learn:
- Robot perception systems using computer vision
- Simultaneous Localization and Mapping (SLAM)
- ROS 2 Navigation Stack configuration
- Path planning algorithms
- Robot manipulation techniques
- State estimation using filters
- Integration of perception, navigation, and manipulation

## Learning Outcomes

By the end of this module, you will be able to:
- Apply AI techniques for robot perception and understanding the environment
- Develop autonomous navigation strategies using ROS 2 Navigation Stack
- Implement basic robot manipulation tasks
- Understand and implement state estimation techniques
- Configure and tune navigation systems
- Plan paths for mobile robots in complex environments

---

## Robot Perception Systems

### Overview

Robot perception is the ability of a robot to understand and interpret its environment through various sensors. This is a fundamental capability that enables robots to navigate, interact with objects, and make intelligent decisions.

### Feature Detection

Feature detection is crucial for robots to identify and track objects in their environment.

#### Key Point Detection

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FeatureDetector(Node):
    def __init__(self):
        super().__init__('feature_detector')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/camera/features', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect features using ORB
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            output_image = cv2.drawKeypoints(cv_image, keypoints, None, color=(0, 255, 0))
            output_msg = self.bridge.cv2_to_imgmsg(output_image, "bgr8")
            self.image_pub.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
```

### Object Recognition

#### Color-Based Object Detection

```python
import cv2
import numpy as np

class ColorDetector:
    def __init__(self):
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([10, 255, 255])

    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image
```

### Point Cloud Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')
        self.pc_sub = self.create_subscription(PointCloud2, '/points', self.pc_callback, 10)

    def pc_callback(self, msg):
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array(points_list)
        if len(points) > 0:
            mean_z = np.mean(points[:, 2])
            self.get_logger().info(f'Point cloud: {len(points)} points')
```

---

## SLAM Implementation

### What is SLAM?

SLAM (Simultaneous Localization and Mapping) is the computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

### Common SLAM Approaches

1. **Graph-Based SLAM**: Represents problem as graph optimization
2. **EKF SLAM**: Uses Extended Kalman Filter for state estimation
3. **Particle Filter SLAM**: Maintains multiple hypotheses about robot location

### ROS 2 Navigation Stack Configuration

The ROS 2 Navigation Stack (Nav2) provides:
- Global and local planners
- Costmap management
- Localization (AMCL)
- Behavior trees for complex behaviors
- Recovery behaviors

### SLAM Toolbox Configuration

```yaml
slam_toolbox:
  ros__parameters:
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    map_update_interval: 2.0
    resolution: 0.05
    max_laser_range: 20.0
    use_scan_matching: true
    do_loop_closing: true
```

### Basic SLAM Node Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import math

class SimpleSLAMNode(Node):
    def __init__(self):
        super().__init__('simple_slam')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.current_pose = [0.0, 0.0, 0.0]
        self.map_resolution = 0.05
        self.map_width = 400
        self.map_height = 400
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        self.occupancy_grid.fill(-1)

        self.map_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        self.current_pose[0] = pose.position.x
        self.current_pose[1] = pose.position.y

    def scan_callback(self, msg):
        robot_map_x = int((self.current_pose[0] + 10) / self.map_resolution)
        robot_map_y = int((self.current_pose[1] + 10) / self.map_resolution)

        angle = msg.angle_min
        for range_val in msg.ranges:
            if not (math.isnan(range_val) or range_val > msg.range_max):
                beam_angle = self.current_pose[2] + angle
                end_x = self.current_pose[0] + range_val * math.cos(beam_angle)
                end_y = self.current_pose[1] + range_val * math.sin(beam_angle)
            angle += msg.angle_increment

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        map_msg.data = self.occupancy_grid.flatten().tolist()
        self.map_pub.publish(map_msg)
```

## Labs

This module includes two hands-on labs:
- **Lab 5**: SLAM and Navigation in an unknown environment
- **Lab 6**: Object detection and grasping with a manipulator

## Acceptance Criteria Met

- [X] Perception techniques with OpenCV examples
- [X] SLAM algorithm explanations
- [X] ROS 2 Navigation Stack configuration
- [X] Practical SLAM implementation examples
