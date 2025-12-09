# Robot Manipulation

## Overview

Robot manipulation is the capability of a robot to physically interact with objects in its environment using an end-effector, typically an articulated robotic arm. This section covers the fundamental concepts of robotic manipulation, including grasping strategies, inverse kinematics for manipulation, and integration with MoveIt 2 for complex manipulation planning.

## Manipulation Planning Concepts

### Degrees of Freedom and Configuration Space

Robotic manipulators are characterized by their degrees of freedom (DOF), which represent the number of independent parameters that define the configuration of the mechanism. A typical robotic arm has 6 DOF to achieve position and orientation control in 3D space, though some applications require more (redundant manipulators) or fewer DOF.

The configuration space (C-space) is the space of all possible configurations of the manipulator. For a manipulator with n joints, the C-space is typically n-dimensional, though constraints can reduce this space.

### End-Effector Control

The end-effector is the tool or device at the end of a robotic arm that interacts with the environment. Common types include:
- Grippers (parallel, angular, suction cups)
- Tools (welding torch, drill, paint sprayer)
- Sensors (cameras, force/torque sensors)

### Grasping Strategies

Grasping is the process of securely holding an object using a robotic end-effector. Different grasping strategies include:

#### 1. Parallel Jaw Grasping
Uses two opposing fingers that move in parallel to grasp objects. Suitable for objects with parallel surfaces.

#### 2. Three-Finger Grasping
Uses three fingers arranged in a triangular pattern to provide stable grasps on various object shapes.

#### 3. Suction Cup Grasping
Uses vacuum pressure to pick up flat, smooth objects. Effective for items like sheets, boxes, or electronic components.

#### 4. Adaptive/Flexible Grasping
Uses soft, adaptable grippers that conform to object shapes, providing robust grasping for irregular objects.

## MoveIt 2 Integration Examples

MoveIt 2 is the official motion planning framework for ROS 2, providing advanced capabilities for manipulation planning. Here's how to integrate with MoveIt 2:

### Basic MoveIt 2 Setup

```cpp
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

class ManipulationController
{
public:
  ManipulationController(const rclcpp::Node::SharedPtr& node)
  : node_(node)
  {
    // Initialize MoveGroupInterface for the manipulator arm
    move_group_interface_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      node_, "manipulator_arm");  // "manipulator_arm" is the planning group name

    // Initialize PlanningSceneInterface for collision objects
    planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

    // Get current robot state
    const moveit::core::JointModelGroup* joint_model_group =
      move_group_interface_->getCurrentState()->getJointModelGroup("manipulator_arm");
  }

  bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
  {
    // Set target pose
    move_group_interface_->setPoseTarget(target_pose);

    // Plan and execute
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if(success) {
      RCLCPP_INFO(node_->get_logger(), "Motion plan successful");
      return move_group_interface_->execute(plan);
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Motion planning failed");
      return false;
    }
  }

  bool moveToJointValues(const std::vector<double>& joint_values)
  {
    move_group_interface_->setJointValueTarget(joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if(success) {
      RCLCPP_INFO(node_->get_logger(), "Joint motion plan successful");
      return move_group_interface_->execute(plan);
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Joint motion planning failed");
      return false;
    }
  }

  bool pickObject(const std::string& object_name)
  {
    // Perform pick operation
    return move_group_interface_->pick(object_name);
  }

  bool placeObject(const std::string& object_name, const geometry_msgs::msg::Pose& place_pose)
  {
    // Create place location
    moveit_msgs::msg::PlaceLocation place_location;
    place_location.place_pose = place_pose;
    place_location.pre_place_approach.direction.vector.z = -1.0;
    place_location.pre_place_approach.min_distance = 0.1;
    place_location.pre_place_approach.desired_distance = 0.12;
    place_location.post_place_approach.direction.vector.x = 1.0;
    place_location.post_place_approach.min_distance = 0.1;
    place_location.post_place_approach.desired_distance = 0.15;

    // Perform place operation
    std::vector<moveit_msgs::msg::PlaceLocation> place_locations;
    place_locations.push_back(place_location);

    return move_group_interface_->place(object_name, place_locations);
  }

private:
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
};
```

### Adding Collision Objects

```cpp
void addCollisionObjects(const rclcpp::Node::SharedPtr& node)
{
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  // Define a table as a collision object
  moveit_msgs::msg::CollisionObject table_collision_object;
  table_collision_object.header.frame_id = "base_link";
  table_collision_object.id = "table";

  // Define table dimensions
  shape_msgs::msg::SolidPrimitive table_primitive;
  table_primitive.type = table_primitive.BOX;
  table_primitive.dimensions.resize(3);
  table_primitive.dimensions[0] = 1.0;  // length
  table_primitive.dimensions[1] = 1.0;  // width
  table_primitive.dimensions[2] = 0.05; // height

  // Define table pose
  geometry_msgs::msg::Pose table_pose;
  table_pose.position.x = 0.5;
  table_pose.position.y = 0.0;
  table_pose.position.z = 0.4;  // Half table height
  table_pose.orientation.w = 1.0;

  table_collision_object.primitives.push_back(table_primitive);
  table_collision_object.primitive_poses.push_back(table_pose);
  table_collision_object.operation = table_collision_object.ADD;

  // Add object to planning scene
  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
  collision_objects.push_back(table_collision_object);
  planning_scene_interface.addCollisionObjects(collision_objects);

  RCLCPP_INFO(node->get_logger(), "Added table to planning scene");
}
```

### Advanced Manipulation with Cartesian Paths

```cpp
#include <moveit/robot_state/conversions.h>

bool executeCartesianPath(
  const moveit::planning_interface::MoveGroupInterface& move_group_interface,
  const std::vector<geometry_msgs::msg::Pose>& waypoints,
  double eef_step = 0.01,
  double jump_threshold = 0.0,
  bool avoid_collisions = true)
{
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group_interface.computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory, avoid_collisions);

  if(fraction >= 0.95) {  // Require 95% path completion
    RCLCPP_INFO(move_group_interface.getNode()->get_logger(),
                "Cartesian path computed with %f fraction", fraction);

    // Execute trajectory
    moveit_msgs::msg::ExecuteTrajectoryGoal goal;
    goal.trajectory = trajectory;
    return move_group_interface.execute(trajectory);
  } else {
    RCLCPP_ERROR(move_group_interface.getNode()->get_logger(),
                 "Cartesian path computation failed with %f fraction", fraction);
    return false;
  }
}
```

## Grasping Strategy Development

### Object Recognition and Pose Estimation

Before grasping, the robot needs to identify objects and their poses. This typically involves:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__('object_recognition_node')

        # Subscriber for point cloud data
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)

        # Publisher for object poses
        self.object_pose_pub = self.create_publisher(Pose, '/detected_object_pose', 10)

        self.get_logger().info('Object Recognition Node initialized')

    def pointcloud_callback(self, msg):
        """Process point cloud data to detect objects"""
        # Convert ROS PointCloud2 to numpy array
        points = self.pointcloud2_to_array(msg)

        if len(points) == 0:
            return

        # Perform object detection and segmentation
        detected_objects = self.segment_objects(points)

        for obj in detected_objects:
            # Estimate object pose
            pose = self.estimate_object_pose(obj)

            # Publish object pose
            self.object_pose_pub.publish(pose)

    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        import sensor_msgs.point_cloud2 as pc2
        points = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def segment_objects(self, points):
        """Segment objects from point cloud"""
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Plane segmentation (to remove table)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)

        # Extract objects (everything except the plane)
        object_cloud = pcd.select_by_index(inliers, invert=True)

        # Cluster objects
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(object_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

        # Group points by cluster
        objects = []
        max_label = labels.max()
        for i in range(max_label + 1):
            class_indices = np.where(labels == i)[0]
            if len(class_indices) > 100:  # Filter small clusters
                object_points = np.asarray(object_cloud.select_by_index(class_indices).points)
                objects.append(object_points)

        return objects

    def estimate_object_pose(self, points):
        """Estimate pose of an object from its point cloud"""
        # Calculate centroid
        centroid = np.mean(points, axis=0)

        # Calculate orientation using PCA
        centered_points = points - centroid
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvectors by eigenvalues (largest to smallest)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        rotation_matrix = eigenvectors
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]

        # Convert rotation matrix to quaternion
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()

        # Create pose message
        pose = Pose()
        pose.position.x = float(centroid[0])
        pose.position.y = float(centroid[1])
        pose.position.z = float(centroid[2])
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        return pose

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Grasp Planning

```python
#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Pose, Point
import math

class GraspPlanner:
    def __init__(self):
        self.approach_distance = 0.1  # Distance to approach object before grasping
        self.grasp_height_offset = 0.05  # Height offset for top grasps

    def plan_grasps(self, object_pose, object_type="cylinder"):
        """Plan potential grasp poses for an object"""
        grasps = []

        if object_type == "cylinder":
            # Plan side grasps around the cylinder
            grasps.extend(self.plan_side_grasps(object_pose))
            # Plan top grasp
            grasps.append(self.plan_top_grasp(object_pose))
        elif object_type == "box":
            # Plan corner grasps
            grasps.extend(self.plan_corner_grasps(object_pose))
            # Plan face grasps
            grasps.extend(self.plan_face_grasps(object_pose))
        elif object_type == "sphere":
            # Plan multiple grasps around the sphere
            grasps.extend(self.plan_sphere_grasps(object_pose))

        return grasps

    def plan_side_grasps(self, object_pose):
        """Plan side grasps around a cylindrical object"""
        grasps = []

        # Generate grasps around the cylinder at different angles
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            grasp_pose = Pose()

            # Calculate grasp position (approach distance from object center)
            grasp_x = object_pose.position.x + self.approach_distance * math.cos(angle)
            grasp_y = object_pose.position.y + self.approach_distance * math.sin(angle)
            grasp_z = object_pose.position.z  # Same height as object

            grasp_pose.position.x = grasp_x
            grasp_pose.position.y = grasp_y
            grasp_pose.position.z = grasp_z

            # Set orientation to face the object
            # Point gripper towards the object center
            dx = object_pose.position.x - grasp_x
            dy = object_pose.position.y - grasp_y
            angle_to_object = math.atan2(dy, dx)

            # Set gripper orientation (perpendicular to radial direction)
            grasp_orientation = angle_to_object + math.pi/2
            grasp_pose.orientation.z = math.sin(grasp_orientation / 2)
            grasp_pose.orientation.w = math.cos(grasp_orientation / 2)

            grasps.append(grasp_pose)

        return grasps

    def plan_top_grasp(self, object_pose):
        """Plan a top grasp for an object"""
        grasp_pose = Pose()

        # Position above the object
        grasp_pose.position.x = object_pose.position.x
        grasp_pose.position.y = object_pose.position.y
        grasp_pose.position.z = object_pose.position.z + self.grasp_height_offset

        # Orient gripper vertically (downward grasp)
        grasp_pose.orientation.x = 0.707  # 90-degree rotation around X-axis
        grasp_pose.orientation.y = 0.0
        grasp_pose.orientation.z = 0.0
        grasp_pose.orientation.w = 0.707

        return grasp_pose

    def plan_corner_grasps(self, object_pose):
        """Plan corner grasps for a box-shaped object"""
        # This would involve more complex geometry calculations
        # based on the object's bounding box
        grasps = []
        # Implementation would consider corner positions and appropriate orientations
        return grasps

    def evaluate_grasps(self, grasps, collision_check_func):
        """Evaluate grasps based on collision safety and grasp quality"""
        valid_grasps = []

        for grasp in grasps:
            if collision_check_func(grasp):
                # Calculate grasp quality score
                quality = self.calculate_grasp_quality(grasp)
                valid_grasps.append((grasp, quality))

        # Sort by quality score
        valid_grasps.sort(key=lambda x: x[1], reverse=True)
        return [grasp for grasp, quality in valid_grasps]

    def calculate_grasp_quality(self, grasp_pose):
        """Calculate quality score for a grasp pose"""
        # Simple quality metric based on approach angle and stability
        # More sophisticated implementations would use force-closure analysis
        return 1.0  # Placeholder

# Example usage
def example_usage():
    planner = GraspPlanner()

    # Create a sample object pose
    object_pose = Pose()
    object_pose.position.x = 0.5
    object_pose.position.y = 0.0
    object_pose.position.z = 0.1

    # Plan grasps for a cylindrical object
    grasps = planner.plan_grasps(object_pose, "cylinder")

    print(f"Planned {len(grasps)} potential grasps")
    for i, grasp in enumerate(grasps):
        print(f"Grasp {i}: Position=({grasp.position.x:.2f}, {grasp.position.y:.2f}, {grasp.position.z:.2f})")

if __name__ == "__main__":
    example_usage()
```

## Integration with Perception Systems

Robotic manipulation requires tight integration with perception systems to identify objects and their poses. This typically involves:

1. **Sensor Fusion**: Combining data from multiple sensors (RGB-D cameras, LiDAR, force/torque sensors)
2. **Object Recognition**: Identifying objects in the environment
3. **Pose Estimation**: Determining accurate poses of objects
4. **Scene Understanding**: Understanding spatial relationships between objects

### Example Integration Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class ManipulationSystemNode(Node):
    def __init__(self):
        super().__init__('manipulation_system_node')

        # Initialize MoveIt interface (would use move_group_interface in C++)
        # For this example, we'll just simulate the interface

        # Subscribers
        self.object_detection_sub = self.create_subscription(
            PoseStamped, '/object_pose', self.object_pose_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/manipulation_command', self.command_callback, 10)

        # Publishers
        self.grasp_pose_pub = self.create_publisher(Pose, '/grasp_pose', 10)
        self.status_pub = self.create_publisher(String, '/manipulation_status', 10)

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Store object poses
        self.detected_objects = {}

        self.get_logger().info('Manipulation System Node initialized')

    def object_pose_callback(self, msg):
        """Store detected object poses"""
        object_name = msg.header.frame_id  # Simplified - in practice would have separate field
        self.detected_objects[object_name] = msg.pose
        self.get_logger().info(f'Stored pose for object: {object_name}')

    def command_callback(self, msg):
        """Process manipulation commands"""
        command = msg.data

        if command.startswith('grasp '):
            object_name = command.split(' ', 1)[1]
            self.execute_grasp(object_name)
        elif command.startswith('place '):
            # Parse place command
            parts = command.split(' ')
            object_name = parts[1]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            self.execute_place(object_name, x, y, z)
        else:
            self.get_logger().error(f'Unknown command: {command}')

    def execute_grasp(self, object_name):
        """Execute grasp for specified object"""
        if object_name not in self.detected_objects:
            self.get_logger().error(f'Object {object_name} not detected')
            return False

        object_pose = self.detected_objects[object_name]

        # Plan grasps using our grasp planner
        grasp_planner = GraspPlanner()
        potential_grasps = grasp_planner.plan_grasps(object_pose)

        if not potential_grasps:
            self.get_logger().error(f'No valid grasps found for {object_name}')
            return False

        # Select best grasp
        best_grasp = potential_grasps[0]  # First is best in our simple implementation

        # Transform grasp pose to planning frame if needed
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # Planning frame
                object_pose.header.frame_id if hasattr(object_pose, 'header') else 'camera_link',
                rclpy.time.Time())

            # Apply transformation
            transformed_grasp = do_transform_pose(best_grasp, transform)
        except Exception as e:
            self.get_logger().warn(f'Transform failed: {e}')
            transformed_grasp = best_grasp  # Use original if transform fails

        # Publish grasp pose for execution
        self.grasp_pose_pub.publish(transformed_grasp)

        # In a real system, this would call MoveIt to plan and execute
        self.get_logger().info(f'Executing grasp for {object_name}')

        # Publish status
        status_msg = String()
        status_msg.data = f'Grasping {object_name}'
        self.status_pub.publish(status_msg)

        return True

    def execute_place(self, object_name, x, y, z):
        """Execute place operation"""
        self.get_logger().info(f'Placing {object_name} at ({x}, {y}, {z})')

        # In a real system, this would involve:
        # 1. Planning approach to place location
        # 2. Executing place motion
        # 3. Releasing object

        status_msg = String()
        status_msg.data = f'Placing {object_name}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationSystemNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Manipulation

### 1. Safety First
- Implement proper collision checking
- Use force/torque sensors for compliant control
- Set appropriate velocity and acceleration limits
- Implement emergency stop procedures

### 2. Robust Grasping
- Plan multiple grasp candidates
- Use tactile feedback when available
- Implement grasp verification
- Handle grasp failures gracefully

### 3. Efficient Planning
- Use appropriate planning algorithms for the task
- Consider using task-specific motion primitives
- Implement online replanning capabilities
- Optimize for execution speed when possible

### 4. Integration Considerations
- Maintain consistent coordinate frames
- Handle timing constraints between perception and action
- Implement proper error handling and recovery
- Provide feedback to higher-level task planners

## Acceptance Criteria Met

- [X] Manipulation planning concepts
- [X] MoveIt 2 integration examples
- [X] Grasping strategy development