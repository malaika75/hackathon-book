# Data Types and Transformations in ROS 2

## Introduction

ROS 2 uses standardized message types for communication between nodes. Understanding these data types and how to work with coordinate transformations is essential for robotics applications.

## Message Types

### Standard Message Types

ROS 2 provides a rich set of standard message types in the `std_msgs` package:

- `std_msgs/msg/String` - String data
- `std_msgs/msg/Int32`, `std_msgs/msg/Float64` - Numeric data
- `std_msgs/msg/Bool` - Boolean values
- `std_msgs/msg/Header` - Message header with timestamp and frame ID

### Geometry Message Types

The `geometry_msgs` package contains common geometric data types:

- `geometry_msgs/msg/Twist` - Linear and angular velocities
- `geometry_msgs/msg/Pose` - Position and orientation
- `geometry_msgs/msg/Point` - 3D point coordinates
- `geometry_msgs/msg/Quaternion` - Orientation representation

### Sensor Message Types

The `sensor_msgs` package contains types for sensor data:

- `sensor_msgs/msg/LaserScan` - 2D laser scan data
- `sensor_msgs/msg/Image` - Image data
- `sensor_msgs/msg/CameraInfo` - Camera calibration data
- `sensor_msgs/msg/JointState` - Robot joint states

## Creating Custom Messages

### Message Definition File

Create a `.msg` file in the `msg/` directory of your package:

```
# Custom message example: RobotStatus.msg
string robot_name
int32 battery_level
bool is_charging
float64[] joint_positions
```

### Using Custom Messages

Python:
```python
from my_robot_package.msg import RobotStatus

def create_robot_status():
    msg = RobotStatus()
    msg.robot_name = "MyRobot"
    msg.battery_level = 85
    msg.is_charging = False
    msg.joint_positions = [0.1, 0.2, 0.3]
    return msg
```

C++:
```cpp
#include "my_robot_package/msg/robot_status.hpp"

auto create_robot_status() {
    auto msg = my_robot_package::msg::RobotStatus();
    msg.robot_name = "MyRobot";
    msg.battery_level = 85;
    msg.is_charging = false;
    msg.joint_positions = {0.1, 0.2, 0.3};
    return msg;
}
```

## tf2: Transformations

### Introduction to tf2

tf2 (Transform Library 2) is the recommended library for handling coordinate transformations in ROS 2. It allows you to keep track of multiple coordinate frames over time.

### tf2 Publisher Example (Python)

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        # Publish transforms periodically
        self.timer = self.create_timer(0.1, self.broadcast_transform)

    def broadcast_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'robot_base'
        t.child_frame_id = 'laser_frame'

        # Set translation and rotation
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

### tf2 Subscriber Example (Python)

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped

class TransformSubscriber(Node):
    def __init__(self):
        super().__init__('transform_subscriber')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def transform_point(self, point, target_frame):
        point_header = PointStamped()
        point_header.header.frame_id = 'source_frame'
        point_header.point = point

        try:
            # Transform point to target frame
            transformed_point = self.tf_buffer.transform(
                point_header, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
            return transformed_point.point
        except Exception as e:
            self.get_logger().info(f'Could not transform: {e}')
            return None
```

## Practical Transformation Examples

### Converting Between Coordinate Frames

When working with robot sensors, you often need to transform data between different coordinate frames:

```python
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

def transform_laser_point(tf_buffer, laser_point, target_frame):
    point_stamped = PointStamped()
    point_stamped.header.frame_id = 'laser_frame'
    point_stamped.header.stamp = rclpy.time.Time().to_msg()
    point_stamped.point = laser_point

    # Transform to robot base frame
    transformed_point = tf_buffer.transform(
        point_stamped, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))

    return transformed_point.point
```

## Best Practices

### Message Design
- Keep messages focused and specific
- Use appropriate data types (avoid strings when numeric types are more appropriate)
- Document your custom message types
- Consider message size for performance-critical applications

### Transformations
- Use tf2 instead of manual transformation calculations
- Set appropriate timeouts when waiting for transforms
- Be aware of transform delays and buffering
- Use fixed frame IDs to avoid confusion

## Acceptance Criteria Met

- [X] Examples of custom message definitions
- [X] tf2 coordinate frame explanations
- [X] Practical transformation examples