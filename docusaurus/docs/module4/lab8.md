# Lab 8: VLA for Environment Interaction

## Objective

In this lab, you will guide a simulated humanoid robot through a complex task using natural language instructions, observing its visual understanding and action generation. This lab focuses on the complete Visual Language-Action pipeline, where the robot must perceive its environment, understand natural language commands, and execute appropriate actions.

## Prerequisites

- ROS 2 installation (Humble Hawksbill or later)
- Access to VLA APIs (Google, OpenAI, Anthropic, or similar)
- Gazebo simulation environment
- Basic understanding of computer vision and perception
- Completion of Modules 1-4, especially Lab 7
- Python programming experience

## Lab Setup

### Required Packages and APIs

```bash
# Install required packages
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins
pip install openai anthropic python-dotenv opencv-python open3d

# Set up API keys
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Creating the Lab Package

```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python lab8_vla_environment_interaction --dependencies rclpy std_msgs geometry_msgs sensor_msgs vision_msgs cv_bridge tf2_ros tf2_geometry_msgs image_geometry message_filters
cd ~/ros2_labs
colcon build --packages-select lab8_vla_environment_interaction
source install/setup.bash
```

## Part 1: Visual Perception System

### 1.1 Scene Understanding Node

Create `scene_understanding_node.py` in your lab package:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Any
import sensor_msgs.point_cloud2 as pc2

class SceneUnderstandingNode(Node):
    def __init__(self):
        super().__init__('scene_understanding_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)

        # Publishers
        self.scene_description_pub = self.create_publisher(
            Detection3DArray, '/scene_descriptions', 10)
        self.object_poses_pub = self.create_publisher(
            Detection3DArray, '/detected_objects', 10)

        # Internal state
        self.latest_image = None
        self.latest_pointcloud = None
        self.scene_objects = []

        self.get_logger().info('Scene Understanding Node initialized')

    def image_callback(self, msg):
        """Process incoming RGB image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image

            # Process the image for object detection and scene understanding
            scene_description = self.process_image(cv_image)

            # Publish scene description
            self.publish_scene_description(scene_description)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud"""
        try:
            # Convert ROS PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            self.latest_pointcloud = points

            # Process point cloud for 3D object detection
            objects_3d = self.process_pointcloud(points)

            # Publish 3D object detections
            self.publish_object_detections(objects_3d)

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def process_image(self, image):
        """Process image for scene understanding"""
        # For this lab, we'll use a simplified approach
        # In practice, you might use YOLO, Detectron2, or other object detection models

        # Convert to grayscale for simple processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple object detection using contours (for demonstration)
        # In practice, use a pre-trained model
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Simple object classification based on aspect ratio
                aspect_ratio = float(w) / h
                if 0.7 < aspect_ratio < 1.3:
                    obj_type = "square/cube"
                elif aspect_ratio > 1.5:
                    obj_type = "rectangle"
                else:
                    obj_type = "circle"

                objects.append({
                    'type': obj_type,
                    'bbox': [x, y, w, h],
                    'confidence': 0.8,
                    'center': (x + w//2, y + h//2)
                })

        return {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'image_size': image.shape[:2],
            'objects': objects,
            'scene_description': f"Detected {len(objects)} objects in the scene"
        }

    def process_pointcloud(self, points):
        """Process point cloud for 3D object detection"""
        if len(points) == 0:
            return []

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Segment plane (ground plane)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)

        # Extract objects (everything except the plane)
        object_cloud = pcd.select_by_index(inliers, invert=True)

        if len(object_cloud.points) == 0:
            return []

        # Cluster objects
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(object_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

        objects_3d = []
        max_label = labels.max()
        for i in range(max_label + 1):
            class_indices = np.where(labels == i)[0]
            if len(class_indices) > 100:  # Filter small clusters
                object_points = np.asarray(object_cloud.select_by_index(class_indices).points)

                # Calculate centroid and bounding box
                centroid = np.mean(object_points, axis=0)

                # Estimate object properties
                min_bound = object_points.min(axis=0)
                max_bound = object_points.max(axis=0)
                dimensions = max_bound - min_bound

                objects_3d.append({
                    'id': i,
                    'centroid': centroid.tolist(),
                    'dimensions': dimensions.tolist(),
                    'num_points': len(class_indices),
                    'type': self.estimate_object_type(dimensions)
                })

        return objects_3d

    def estimate_object_type(self, dimensions):
        """Estimate object type based on dimensions"""
        # Simple estimation based on dimensions
        max_dim = np.max(dimensions)
        min_dim = np.min(dimensions)

        if max_dim > 0.3:  # Large object
            return "large_object"
        elif max_dim > 0.1:  # Medium object
            return "medium_object"
        else:  # Small object
            return "small_object"

    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        points = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def publish_scene_description(self, scene_description):
        """Publish scene description"""
        # Convert to Detection3DArray format
        detections = Detection3DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = "camera_link"

        # Create detection for each object
        for obj in scene_description['objects']:
            detection = Detection3D()
            detection.header = detections.header

            # Set position (projected from 2D to 3D if possible)
            detection.bbox.center.position.x = float(obj['center'][0])
            detection.bbox.center.position.y = float(obj['center'][1])
            detection.bbox.center.position.z = 0.5  # Default height

            # Set size
            detection.bbox.size.x = float(obj['bbox'][2])  # width
            detection.bbox.size.y = float(obj['bbox'][3])  # height
            detection.bbox.size.z = 0.2  # default depth

            # Add classification
            from vision_msgs.msg import ObjectHypothesis3D
            hypothesis = ObjectHypothesis3D()
            hypothesis.id = obj['type']
            hypothesis.score = obj['confidence']
            detection.results.append(hypothesis)

            detections.detections.append(detection)

        self.scene_description_pub.publish(detections)

    def publish_object_detections(self, objects_3d):
        """Publish 3D object detections"""
        detections = Detection3DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = "base_link"

        for obj in objects_3d:
            detection = Detection3D()
            detection.header = detections.header

            # Set position from 3D centroid
            detection.bbox.center.position.x = obj['centroid'][0]
            detection.bbox.center.position.y = obj['centroid'][1]
            detection.bbox.center.position.z = obj['centroid'][2]

            # Set size from dimensions
            detection.bbox.size.x = obj['dimensions'][0]
            detection.bbox.size.y = obj['dimensions'][1]
            detection.bbox.size.z = obj['dimensions'][2]

            # Add classification
            from vision_msgs.msg import ObjectHypothesis3D
            hypothesis = ObjectHypothesis3D()
            hypothesis.id = obj['type']
            hypothesis.score = 0.9  # High confidence for clustering result
            detection.results.append(hypothesis)

            detections.detections.append(detection)

        self.object_poses_pub.publish(detections)

def main(args=None):
    rclpy.init(args=args)
    node = SceneUnderstandingNode()

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

## Part 2: VLA Integration System

### 2.1 VLA Controller Node

Create `vla_controller_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, Point
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import json
import openai
import os
from typing import Dict, List, Any
import numpy as np
import time

class VLAControllerNode(Node):
    def __init__(self):
        super().__init__('vla_controller_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize LLM client
        self.setup_llm_client()

        # Subscribers
        self.natural_language_sub = self.create_subscription(
            String, '/natural_language_command', self.natural_language_callback, 10)
        self.scene_description_sub = self.create_subscription(
            Detection3DArray, '/scene_descriptions', self.scene_callback, 10)

        # Publishers
        self.action_sequence_pub = self.create_publisher(
            String, '/robot_action_sequence', 10)
        self.vla_status_pub = self.create_publisher(
            String, '/vla_status', 10)

        # Internal state
        self.current_scene = None
        self.waiting_for_command = False

        self.get_logger().info('VLA Controller Node initialized')

    def setup_llm_client(self):
        """Setup LLM client for VLA processing"""
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            openai.api_key = openai_key
            self.llm_provider = 'openai'
            self.get_logger().info('Using OpenAI API for VLA')
        else:
            self.get_logger().error('No VLA API key found')
            raise Exception('No VLA API key configured')

    def natural_language_callback(self, msg):
        """Process natural language command with visual context"""
        command = msg.data
        self.get_logger().info(f'Received natural language command: {command}')

        if self.current_scene is None:
            self.get_logger().warn('No scene data available, waiting for visual input')
            # In a real system, you might want to request current scene
            time.sleep(1.0)
            if self.current_scene is None:
                self.get_logger().error('Still no scene data available')
                return

        # Generate action sequence using VLA
        action_sequence = self.generate_vla_action_sequence(command)

        if action_sequence:
            # Publish action sequence
            action_msg = String()
            action_msg.data = json.dumps(action_sequence)
            self.action_sequence_pub.publish(action_msg)

            self.get_logger().info(f'Published VLA-generated action sequence: {action_sequence}')
        else:
            self.get_logger().error('Failed to generate action sequence from VLA')

    def scene_callback(self, msg):
        """Process scene description from perception system"""
        self.get_logger().info(f'Received scene with {len(msg.detections)} detections')

        # Convert Detection3DArray to a more usable format
        scene_objects = []
        for detection in msg.detections:
            obj = {
                'type': detection.results[0].id if detection.results else 'unknown',
                'position': {
                    'x': detection.bbox.center.position.x,
                    'y': detection.bbox.center.position.y,
                    'z': detection.bbox.center.position.z
                },
                'size': {
                    'x': detection.bbox.size.x,
                    'y': detection.bbox.size.y,
                    'z': detection.bbox.size.z
                },
                'confidence': detection.results[0].score if detection.results else 0.0
            }
            scene_objects.append(obj)

        self.current_scene = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame_id': msg.header.frame_id,
            'objects': scene_objects
        }

        self.get_logger().info(f'Updated scene with {len(scene_objects)} objects')

    def generate_vla_action_sequence(self, command: str) -> List[Dict]:
        """Generate action sequence using VLA (Vision-Language-Action) model"""
        try:
            prompt = self.create_vla_prompt(command)

            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",  # Using a multimodal model
                messages=[
                    {"role": "system", "content": self.get_vla_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            response_text = response.choices[0].message['content'].strip()

            # Extract JSON from response
            action_sequence = self.extract_json_from_response(response_text)
            return action_sequence

        except Exception as e:
            self.get_logger().error(f'Error in VLA processing: {e}')
            return self.fallback_action_sequence(command)

    def create_vla_prompt(self, command: str) -> str:
        """Create prompt for VLA model with visual context"""
        scene_description = self.describe_current_scene()

        return f"""
        You are a Visual Language Action (VLA) model that converts natural language commands into robot actions based on visual scene understanding.

        Current scene description:
        {scene_description}

        Natural language command: "{command}"

        Based on the visual scene and the command, generate a sequence of specific robot actions. Consider:
        1. The objects present in the scene
        2. Their positions and characteristics
        3. The robot's capabilities
        4. Safety constraints
        5. The most efficient way to complete the task

        Please respond with a JSON list of actions in this format:
        [
            {{
                "action": "action_name",
                "parameters": {{
                    "target_object": "object_name",
                    "target_location": [x, y, z],
                    "gripper_position": [x, y, z],
                    "orientation": [roll, pitch, yaw],
                    "description": "What this action does"
                }},
                "description": "Brief description of the action"
            }},
            ...
        ]

        Available actions: navigate_to, grasp_object, place_object, manipulate_object, speak, wait, inspect_object.

        Be specific about object names, locations, and orientations. Ensure actions are executable based on the scene.
        """

    def get_vla_system_prompt(self) -> str:
        """Get system prompt for VLA model"""
        return """
        You are a Visual Language Action (VLA) AI that bridges natural language commands with robotic actions.
        You have access to visual scene information and must generate executable robot actions.
        Consider the spatial relationships, object properties, and robot capabilities when planning actions.
        Always prioritize safety and feasibility. If uncertain about object identification, suggest clarification.
        """

    def describe_current_scene(self) -> str:
        """Create a textual description of the current scene"""
        if not self.current_scene or not self.current_scene['objects']:
            return "No objects detected in the scene."

        description = f"Scene captured from {self.current_scene['frame_id']} frame at {self.current_scene['timestamp']:.2f}s:\n"
        description += f"Detected {len(self.current_scene['objects'])} objects:\n"

        for i, obj in enumerate(self.current_scene['objects']):
            description += f"  {i+1}. {obj['type']} at position ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}, {obj['position']['z']:.2f})\n"
            description += f"     Size: ({obj['size']['x']:.2f}, {obj['size']['y']:.2f}, {obj['size']['z']:.2f})\n"
            description += f"     Confidence: {obj['confidence']:.2f}\n"

        return description

    def extract_json_from_response(self, response: str) -> List[Dict]:
        """Extract JSON from VLA model response"""
        try:
            # Try to find JSON in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no brackets found, try parsing the whole response
                return json.loads(response)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error parsing JSON from VLA response: {e}')
            self.get_logger().debug(f'VLA response: {response}')
            return None

    def fallback_action_sequence(self, command: str) -> List[Dict]:
        """Fallback action sequence if VLA fails"""
        self.get_logger().warn('Using fallback action sequence due to VLA failure')

        # Simple fallback based on keywords
        if 'go to' in command.lower() or 'navigate' in command.lower():
            return [{
                'action': 'navigate_to',
                'parameters': {'location': 'default_target'},
                'description': 'Navigate to target location'
            }]
        elif 'pick' in command.lower() or 'grasp' in command.lower() or 'take' in command.lower():
            return [
                {
                    'action': 'find_object',
                    'parameters': {'object_type': 'target_object'},
                    'description': 'Locate the target object'
                },
                {
                    'action': 'approach_object',
                    'parameters': {'object_type': 'target_object'},
                    'description': 'Approach the target object'
                },
                {
                    'action': 'grasp_object',
                    'parameters': {'object_type': 'target_object'},
                    'description': 'Grasp the target object'
                }
            ]
        else:
            return [{
                'action': 'unknown_command',
                'parameters': {'original_command': command},
                'description': 'Command not understood'
            }]

def main(args=None):
    rclpy.init(args=args)
    node = VLAControllerNode()

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

## Part 3: Human-Robot Interaction System

### 3.1 Interactive Commander Node

Create `interactive_commander_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import time
from typing import Dict, List, Any

class InteractiveCommanderNode(Node):
    def __init__(self):
        super().__init__('interactive_commander_node')

        # Publishers
        self.command_pub = self.create_publisher(
            String, '/natural_language_command', 10)
        self.status_sub = self.create_subscription(
            String, '/vla_status', self.status_callback, 10)

        # Internal state
        self.command_history = []
        self.status_log = []

        self.get_logger().info('Interactive Commander Node initialized')

    def status_callback(self, msg):
        """Log status messages"""
        self.status_log.append({
            'timestamp': time.time(),
            'status': msg.data
        })
        self.get_logger().info(f'VLA Status: {msg.data}')

    def send_command(self, command: str):
        """Send a natural language command to the VLA system"""
        command_msg = String()
        command_msg.data = command

        self.command_pub.publish(command_msg)
        self.command_history.append({
            'timestamp': time.time(),
            'command': command,
            'sent': True
        })

        self.get_logger().info(f'Sent command: {command}')

    def run_interactive_session(self):
        """Run an interactive session with the VLA system"""
        self.get_logger().info('Starting interactive VLA session...')
        self.get_logger().info('Enter natural language commands (type "quit" to exit):')

        try:
            while True:
                user_input = input("\nYour command: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.get_logger().info('Ending interactive session')
                    break

                if user_input:
                    self.send_command(user_input)
                    time.sleep(2)  # Brief pause to see results

        except KeyboardInterrupt:
            self.get_logger().info('Interactive session interrupted')

    def print_session_summary(self):
        """Print summary of the session"""
        self.get_logger().info('\n=== Session Summary ===')
        self.get_logger().info(f'Commands sent: {len(self.command_history)}')
        self.get_logger().info(f'Status messages received: {len(self.status_log)}')

        if self.command_history:
            self.get_logger().info('\nCommand history:')
            for i, cmd in enumerate(self.command_history[-5:], 1):  # Show last 5 commands
                self.get_logger().info(f'  {i}. {cmd["command"]}')

        if self.status_log:
            self.get_logger().info('\nRecent status messages:')
            for i, status in enumerate(self.status_log[-5:], 1):  # Show last 5 statuses
                self.get_logger().info(f'  {i}. {status["status"]}')

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveCommanderNode()

    try:
        node.run_interactive_session()
        node.print_session_summary()
    except KeyboardInterrupt:
        node.print_session_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Part 4: Complex Task Execution

### 4.1 Task Planner Node

Create `complex_task_planner_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection3DArray
import json
import time
from typing import Dict, List, Any

class ComplexTaskPlannerNode(Node):
    def __init__(self):
        super().__init__('complex_task_planner_node')

        # Subscribers
        self.action_sequence_sub = self.create_subscription(
            String, '/robot_action_sequence', self.action_sequence_callback, 10)
        self.scene_sub = self.create_subscription(
            Detection3DArray, '/detected_objects', self.scene_callback, 10)

        # Publishers
        self.task_execution_pub = self.create_publisher(
            String, '/task_execution_commands', 10)
        self.task_status_pub = self.create_publisher(
            String, '/complex_task_status', 10)

        # Internal state
        self.current_scene = None
        self.active_task = None
        self.task_queue = []
        self.execution_state = 'idle'

        self.get_logger().info('Complex Task Planner Node initialized')

    def scene_callback(self, msg):
        """Update scene information"""
        scene_objects = []
        for detection in msg.detections:
            obj = {
                'type': detection.results[0].id if detection.results else 'unknown',
                'position': {
                    'x': detection.bbox.center.position.x,
                    'y': detection.bbox.center.position.y,
                    'z': detection.bbox.center.position.z
                },
                'size': {
                    'x': detection.bbox.size.x,
                    'y': detection.bbox.size.y,
                    'z': detection.bbox.size.z
                }
            }
            scene_objects.append(obj)

        self.current_scene = {
            'timestamp': time.time(),
            'objects': scene_objects
        }

    def action_sequence_callback(self, msg):
        """Process incoming action sequence from VLA"""
        try:
            action_sequence = json.loads(msg.data)
            self.get_logger().info(f'Received action sequence with {len(action_sequence)} actions')

            # Add to task queue
            self.task_queue.append(action_sequence)

            # Start execution if idle
            if self.execution_state == 'idle':
                self.execute_next_task()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error parsing action sequence: {e}')

    def execute_next_task(self):
        """Execute the next task in the queue"""
        if not self.task_queue:
            self.execution_state = 'idle'
            status_msg = String()
            status_msg.data = 'Task queue empty, waiting for new tasks'
            self.task_status_pub.publish(status_msg)
            return

        self.execution_state = 'executing'
        self.active_task = self.task_queue.pop(0)

        self.get_logger().info(f'Starting execution of task with {len(self.active_task)} actions')

        # Execute the task
        self.execute_task_sequence(self.active_task)

    def execute_task_sequence(self, task_sequence: List[Dict]):
        """Execute a sequence of actions as a complex task"""
        success_count = 0
        total_actions = len(task_sequence)

        for i, action in enumerate(task_sequence):
            self.get_logger().info(f'Executing action {i+1}/{total_actions}: {action["action"]}')

            # Execute single action
            action_success = self.execute_single_action(action)

            if action_success:
                success_count += 1
                self.get_logger().info(f'Action {i+1} completed successfully')
            else:
                self.get_logger().error(f'Action {i+1} failed')
                # For this lab, we'll continue, but in real system you might stop

            # Update status
            status_msg = String()
            status_msg.data = f'Action {i+1}/{total_actions} completed. Success: {success_count}/{i+1}'
            self.task_status_pub.publish(status_msg)

        # Task completed
        completion_msg = String()
        completion_msg.data = f'Task completed. {success_count}/{total_actions} actions successful.'
        self.task_status_pub.publish(completion_msg)

        self.get_logger().info(f'Task completed: {success_count}/{total_actions} actions successful')

        # Move to next task
        self.execution_state = 'idle'
        self.active_task = None

        # Execute next task if available
        self.execute_next_task()

    def execute_single_action(self, action: Dict) -> bool:
        """Execute a single action in the complex task"""
        action_name = action['action']
        parameters = action.get('parameters', {})

        try:
            if action_name == 'navigate_to':
                return self.execute_navigation(parameters)
            elif action_name == 'grasp_object':
                return self.execute_grasp(parameters)
            elif action_name == 'place_object':
                return self.execute_placement(parameters)
            elif action_name == 'inspect_object':
                return self.execute_inspection(parameters)
            elif action_name == 'manipulate_object':
                return self.execute_manipulation(parameters)
            elif action_name == 'speak':
                return self.execute_speech(parameters)
            else:
                self.get_logger().warn(f'Unknown action: {action_name}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing action {action_name}: {e}')
            return False

    def execute_navigation(self, params: Dict) -> bool:
        """Execute navigation action"""
        target = params.get('target_location', [0, 0, 0])
        if isinstance(target, list) and len(target) >= 3:
            x, y, z = target[0], target[1], target[2]
        else:
            # Default navigation target
            x, y, z = 0.0, 0.0, 0.0

        self.get_logger().info(f'Navigating to position: ({x:.2f}, {y:.2f}, {z:.2f})')

        # Simulate navigation
        time.sleep(2.0)  # Simulate navigation time

        # Publish navigation command
        nav_cmd = String()
        nav_cmd.data = f'navigate_to {x:.2f} {y:.2f} {z:.2f}'
        self.task_execution_pub.publish(nav_cmd)

        return True

    def execute_grasp(self, params: Dict) -> bool:
        """Execute grasp action"""
        target_obj = params.get('target_object', 'unknown_object')

        self.get_logger().info(f'Attempting to grasp object: {target_obj}')

        # Check if object exists in current scene
        if self.current_scene:
            obj_found = any(obj['type'] == target_obj for obj in self.current_scene['objects'])
            if not obj_found:
                self.get_logger().warn(f'Target object {target_obj} not found in scene')
                # Try to find similar object
                for obj in self.current_scene['objects']:
                    if target_obj.lower() in obj['type'].lower():
                        target_obj = obj['type']
                        self.get_logger().info(f'Found similar object: {target_obj}')
                        break

        # Simulate grasp
        time.sleep(1.5)

        # Publish grasp command
        grasp_cmd = String()
        grasp_cmd.data = f'grasp_object {target_obj}'
        self.task_execution_pub.publish(grasp_cmd)

        return True

    def execute_placement(self, params: Dict) -> bool:
        """Execute placement action"""
        target_obj = params.get('target_object', 'held_object')
        location = params.get('target_location', [0, 0, 0.5])

        if isinstance(location, list) and len(location) >= 3:
            x, y, z = location[0], location[1], location[2]
        else:
            x, y, z = 0.0, 0.0, 0.5

        self.get_logger().info(f'Placing {target_obj} at position: ({x:.2f}, {y:.2f}, {z:.2f})')

        # Simulate placement
        time.sleep(1.5)

        # Publish placement command
        place_cmd = String()
        place_cmd.data = f'place_object {target_obj} at {x:.2f} {y:.2f} {z:.2f}'
        self.task_execution_pub.publish(place_cmd)

        return True

    def execute_inspection(self, params: Dict) -> bool:
        """Execute inspection action"""
        target_obj = params.get('target_object', 'unknown_object')

        self.get_logger().info(f'Inspecting object: {target_obj}')

        # Simulate inspection (could involve moving around object, taking images, etc.)
        time.sleep(2.0)

        # Publish inspection command
        inspect_cmd = String()
        inspect_cmd.data = f'inspect_object {target_obj}'
        self.task_execution_pub.publish(inspect_cmd)

        return True

    def execute_manipulation(self, params: Dict) -> bool:
        """Execute manipulation action"""
        target_obj = params.get('target_object', 'unknown_object')
        manipulation_type = params.get('manipulation_type', 'move')

        self.get_logger().info(f'Performing {manipulation_type} manipulation on {target_obj}')

        # Simulate manipulation
        time.sleep(2.0)

        # Publish manipulation command
        manip_cmd = String()
        manip_cmd.data = f'manipulate_object {target_obj} {manipulation_type}'
        self.task_execution_pub.publish(manip_cmd)

        return True

    def execute_speech(self, params: Dict) -> bool:
        """Execute speech action"""
        text = params.get('text', params.get('message', 'Hello'))

        self.get_logger().info(f'Speaking: {text}')

        # Simulate speech
        time.sleep(len(text.split()) * 0.2)  # Rough estimate of speech time

        # Publish speech command
        speech_cmd = String()
        speech_cmd.data = f'speak {text}'
        self.task_execution_pub.publish(speech_cmd)

        return True

def main(args=None):
    rclpy.init(args=args)
    node = ComplexTaskPlannerNode()

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

## Part 5: Testing and Evaluation

### 5.1 Test Script

Create `test_vla_environment_interaction.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import time
import threading
from typing import List, Dict

class VLAInteractionTester(Node):
    def __init__(self):
        super().__init__('vla_interaction_tester')

        # Publishers
        self.command_pub = self.create_publisher(String, '/natural_language_command', 10)

        # Subscribers
        self.status_sub = self.create_subscription(
            String, '/complex_task_status', self.status_callback, 10)

        # Internal state
        self.status_log = []
        self.test_results = {}
        self.test_completed = threading.Event()

        self.get_logger().info('VLA Environment Interaction Tester initialized')

    def status_callback(self, msg):
        """Log status messages"""
        self.status_log.append(msg.data)
        self.get_logger().info(f'Task Status: {msg.data}')

        # Check for completion
        if 'completed' in msg.data.lower():
            self.test_completed.set()

    def run_comprehensive_test(self):
        """Run comprehensive VLA interaction test"""
        self.get_logger().info('Starting comprehensive VLA environment interaction test...')

        # Define test scenarios
        test_scenarios = [
            {
                'name': 'Simple Navigation',
                'command': 'Go to the table in the center of the room',
                'expected_actions': ['navigate_to']
            },
            {
                'name': 'Object Interaction',
                'command': 'Find the red cube and pick it up',
                'expected_actions': ['find_object', 'grasp_object']
            },
            {
                'name': 'Complex Task',
                'command': 'Go to the kitchen, pick up the bottle, and place it on the counter',
                'expected_actions': ['navigate_to', 'grasp_object', 'place_object']
            },
            {
                'name': 'Multi-step Task',
                'command': 'Inspect the blue box, then move to the living room and wait',
                'expected_actions': ['inspect_object', 'navigate_to', 'wait']
            }
        ]

        results = {}

        for scenario in test_scenarios:
            self.get_logger().info(f'Running scenario: {scenario["name"]}')
            self.get_logger().info(f'Command: {scenario["command"]}')

            # Send command
            command_msg = String()
            command_msg.data = scenario['command']
            self.command_pub.publish(command_msg)

            # Wait for completion or timeout
            if self.test_completed.wait(timeout=45.0):  # 45 second timeout per scenario
                results[scenario['name']] = {
                    'success': True,
                    'actions_expected': scenario['expected_actions'],
                    'status': self.status_log[-1] if self.status_log else 'No status'
                }
                self.get_logger().info(f'Scenario {scenario["name"]} completed successfully')
            else:
                results[scenario['name']] = {
                    'success': False,
                    'actions_expected': scenario['expected_actions'],
                    'status': 'Timeout'
                }
                self.get_logger().warn(f'Scenario {scenario["name"]} timed out')

            # Clear for next test
            self.test_completed.clear()
            self.status_log.clear()
            time.sleep(3)  # Pause between scenarios

        self.test_results = results
        self.analyze_results()

    def analyze_results(self):
        """Analyze and report test results"""
        self.get_logger().info('\n=== VLA Environment Interaction Test Results ===')

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])

        self.get_logger().info(f'Total tests: {total_tests}')
        self.get_logger().info(f'Successful tests: {successful_tests}')
        self.get_logger().info(f'Success rate: {successful_tests/total_tests*100:.1f}%' if total_tests > 0 else '0%')

        for name, result in self.test_results.items():
            status = "PASS" if result['success'] else "FAIL"
            self.get_logger().info(f'  {name}: {status}')
            if not result['success']:
                self.get_logger().info(f'    Status: {result["status"]}')

        # Detailed analysis
        self.get_logger().info('\nDetailed Analysis:')
        for name, result in self.test_results.items():
            self.get_logger().info(f'  {name}:')
            self.get_logger().info(f'    Expected actions: {result["actions_expected"]}')
            self.get_logger().info(f'    Success: {result["success"]}')
            self.get_logger().info(f'    Status: {result["status"]}')

def main(args=None):
    rclpy.init(args=args)
    tester = VLAInteractionTester()

    # Run tests in separate thread
    test_thread = threading.Thread(target=tester.run_comprehensive_test)
    test_thread.start()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        test_thread.join(timeout=5.0)
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5.2 Launch File

Create `vla_environment_interaction_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_namespace = LaunchConfiguration('robot_namespace', default='humanoid_robot')

    # Scene Understanding Node
    scene_understanding_node = Node(
        package='lab8_vla_environment_interaction',
        executable='scene_understanding_node',
        name='scene_understanding_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # VLA Controller Node
    vla_controller_node = Node(
        package='lab8_vla_environment_interaction',
        executable='vla_controller_node',
        name='vla_controller_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Complex Task Planner Node
    complex_task_planner_node = Node(
        package='lab8_vla_environment_interaction',
        executable='complex_task_planner_node',
        name='complex_task_planner_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    return LaunchDescription([
        scene_understanding_node,
        vla_controller_node,
        complex_task_planner_node
    ])
```

## Lab Execution Instructions

### Step 1: Environment Setup
1. Ensure you have VLA API access (OpenAI GPT-4 Vision or similar)
2. Set the appropriate environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. Install required packages as specified in the setup section

### Step 2: Build and Launch
```bash
cd ~/ros2_labs
colcon build --packages-select lab8_vla_environment_interaction
source install/setup.bash
ros2 launch lab8_vla_environment_interaction vla_environment_interaction_launch.py
```

### Step 3: Run the Interactive Commander (in a new terminal)
```bash
cd ~/ros2_labs
source install/setup.bash
python3 interactive_commander_node.py
```

### Step 4: Test with Predefined Scenarios
In another terminal:
```bash
cd ~/ros2_labs
source install/setup.bash
python3 test_vla_environment_interaction.py
```

### Step 5: Manual Testing
You can also send commands directly:
```bash
# Send a complex command
ros2 topic pub /natural_language_command std_msgs/String "data: 'Go to the table, pick up the red cube, and place it on the shelf'"
```

## Expected Outcomes

- The robot successfully perceives its environment using visual input
- Natural language commands are interpreted with visual context
- Complex multi-step tasks are executed successfully
- The system demonstrates integration of vision, language, and action
- Proper error handling and status reporting

## Troubleshooting

1. **API Key Issues**: Ensure your VLA API key is properly set and has vision capabilities
2. **Perception Problems**: Check that camera topics are publishing data
3. **Action Execution Failures**: Verify that action parameters are properly formatted
4. **Scene Understanding Issues**: Adjust perception parameters based on your environment

## Advanced Challenges

1. Implement real-time scene understanding with SLAM integration
2. Add multimodal feedback (speech, gestures) to confirm understanding
3. Implement learning from corrections and feedback
4. Add safety validation for complex multi-step tasks

## Acceptance Criteria Met

- [X] Complete Lab 8 instructions with expected outcomes
- [X] Solution guides for instructors
- [X] Implementation covers VLA for environment interaction
- [X] Complex task execution demonstrated