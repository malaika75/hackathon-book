# Robot Perception Systems

## Overview

Robot perception is the ability of a robot to understand and interpret its environment through various sensors. This is a fundamental capability that enables robots to navigate, interact with objects, and make intelligent decisions based on sensory input.

## Computer Vision for Robotics

### Introduction to Robot Perception

Robot perception encompasses several key capabilities:
- Object detection and recognition
- Environment mapping
- Localization
- Scene understanding
- Sensor fusion

### Feature Detection

Feature detection is crucial for robots to identify and track objects in their environment.

#### Key Point Detection

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class FeatureDetector(Node):
    def __init__(self):
        super().__init__('feature_detector')

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publisher for processed images
        self.image_pub = self.create_publisher(
            Image, '/camera/features', 10)

        # CV Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()

        self.get_logger().info('Feature Detector initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(cv_image, "bgr8", cv2.COLOR_BGR2GRAY)

            # Detect features using ORB
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            # Draw keypoints on the image
            output_image = cv2.drawKeypoints(
                cv_image, keypoints, None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Convert back to ROS Image message
            output_msg = self.bridge.cv2_to_imgmsg(output_image, "bgr8")
            output_msg.header = msg.header

            # Publish the processed image
            self.image_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    feature_detector = FeatureDetector()

    try:
        rclpy.spin(feature_detector)
    except KeyboardInterrupt:
        pass
    finally:
        feature_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### SIFT and SURF Features

For more advanced feature detection, SIFT and SURF algorithms can be used (though SIFT is patented and may require licensing):

```python
# SIFT feature detection (if available in your OpenCV build)
def detect_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints
    output_image = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image, keypoints, descriptors
```

### Object Recognition

#### Template Matching

Template matching is a simple but effective method for object recognition:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TemplateMatcher(Node):
    def __init__(self):
        super().__init__('template_matcher')

        # Load template image
        self.template = cv2.imread('path/to/template.jpg', 0)
        if self.template is None:
            self.get_logger().error('Could not load template image')
            return

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(
            Image, '/camera/match_result', 10)

        self.bridge = CvBridge()

        self.get_logger().info('Template Matcher initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(
                gray, self.template, cv2.TM_CCOEFF_NORMED)

            # Find locations where matching exceeds threshold
            threshold = 0.8
            locations = np.where(result >= threshold)

            # Draw rectangles around matched regions
            h, w = self.template.shape
            for pt in zip(*locations[::-1]):
                cv2.rectangle(
                    cv_image, pt,
                    (pt[0] + w, pt[1] + h),
                    (0, 255, 0), 2)

            # Convert back to ROS Image
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error in template matching: {e}')

def main(args=None):
    rclpy.init(args=args)
    template_matcher = TemplateMatcher()

    try:
        rclpy.spin(template_matcher)
    except KeyboardInterrupt:
        pass
    finally:
        template_matcher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Color-Based Object Detection

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(
            Image, '/camera/color_detection', 10)

        self.bridge = CvBridge()

        # Define color range (HSV format)
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])

        self.get_logger().info('Color Detector initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert BGR to HSV
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Create masks for red color (red wraps around in HSV)
            mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = mask1 + mask2

            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around detected objects
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(
                        cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert back to ROS Image
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error in color detection: {e}')

def main(args=None):
    rclpy.init(args=args)
    color_detector = ColorDetector()

    try:
        rclpy.spin(color_detector)
    except KeyboardInterrupt:
        pass
    finally:
        color_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## OpenCV Integration Examples

### Basic Image Processing Pipeline

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.image_pub = self.create_publisher(
            Image, '/camera/processed', 10)

        self.bridge = CvBridge()
        self.camera_info = None

        self.get_logger().info('Image Processor initialized')

    def info_callback(self, msg):
        """Store camera calibration info"""
        self.camera_info = msg

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)

            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Define range for blue color
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])

            # Create mask for blue color
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Apply mask to original image
            result = cv2.bitwise_and(cv_image, cv_image, mask=mask)

            # Find contours and draw bounding boxes
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(
                        result, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert back to ROS Image
            output_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()

    try:
        rclpy.spin(image_processor)
    except KeyboardInterrupt:
        pass
    finally:
        image_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Point Cloud Processing Basics

### Working with Point Clouds

Point clouds provide 3D information about the environment, which is crucial for robotics applications:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')

        self.pc_sub = self.create_subscription(
            PointCloud2, '/points', self.pc_callback, 10)

        self.get_logger().info('Point Cloud Processor initialized')

    def pc_callback(self, msg):
        """Process point cloud data"""
        try:
            # Convert PointCloud2 to list of points
            points_list = list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True))

            # Convert to numpy array for processing
            points = np.array(points_list)

            if len(points) > 0:
                # Calculate basic statistics
                mean_z = np.mean(points[:, 2])  # Average Z value
                std_z = np.std(points[:, 2])    # Standard deviation of Z

                # Filter points based on Z value (e.g., ground plane removal)
                filtered_points = points[points[:, 2] > mean_z - 2*std_z]

                self.get_logger().info(
                    f'Point cloud: {len(points)} points, '
                    f'filtered: {len(filtered_points)} points')

                # Here you could implement more complex processing
                # like plane fitting, clustering, etc.

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

def main(args=None):
    rclpy.init(args=args)
    pc_processor = PointCloudProcessor()

    try:
        rclpy.spin(pc_processor)
    except KeyboardInterrupt:
        pass
    finally:
        pc_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Perception Integration with ROS 2

### Perception Pipeline Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

class PerceptionPipeline(Node):
    def __init__(self):
        super().__init__('perception_pipeline')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        # Publications
        self.detection_pub = self.create_publisher(
            Detection2D, '/detections', 10)
        self.image_pub = self.create_publisher(
            Image, '/camera/perception_output', 10)

        self.bridge = CvBridge()
        self.camera_info = None

        self.get_logger().info('Perception Pipeline initialized')

    def info_callback(self, msg):
        """Store camera calibration info"""
        self.camera_info = msg

    def detect_objects(self, image):
        """Detect objects in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use a pre-trained Haar cascade classifier for face detection
        # In practice, you'd use more sophisticated methods like YOLO or SSD
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detections = []
        for (x, y, w, h) in faces:
            detection = Detection2D()
            detection.header.stamp = self.get_clock().now().to_msg()
            detection.header.frame_id = 'camera_frame'

            # Set bounding box
            detection.bbox.center.x = x + w/2
            detection.bbox.center.y = y + h/2
            detection.bbox.size_x = w
            detection.bbox.size_y = h

            # Set confidence and class
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'face'
            hypothesis.hypothesis.score = 0.9
            detection.results.append(hypothesis)

            detections.append(detection)

        return detections, faces

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection
            detections, bounding_boxes = self.detect_objects(cv_image)

            # Draw bounding boxes on image
            output_image = cv_image.copy()
            for (x, y, w, h) in bounding_boxes:
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Publish detections
            for detection in detections:
                self.detection_pub.publish(detection)

            # Publish processed image
            output_msg = self.bridge.cv2_to_imgmsg(output_image, "bgr8")
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error in perception pipeline: {e}')

def main(args=None):
    rclpy.init(args=args)
    perception_pipeline = PerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Robot Perception

### Performance Considerations

1. **Efficient Processing**: Use appropriate image resolution and processing frequency
2. **Hardware Acceleration**: Leverage GPU processing when available
3. **Multi-threading**: Separate perception from control loops when possible
4. **Optimized Algorithms**: Choose algorithms that balance accuracy and speed

### Accuracy Considerations

1. **Lighting Conditions**: Account for varying lighting in your algorithms
2. **Sensor Calibration**: Properly calibrate cameras and other sensors
3. **Validation**: Test perception systems under various conditions
4. **Redundancy**: Use multiple sensors when possible for robustness

## Integration with Other Systems

### Perception-Action Loop

Robot perception should be tightly integrated with action systems:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2D
import cv2

class PerceptionActionNode(Node):
    def __init__(self):
        super().__init__('perception_action')

        # Perception
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.detection_sub = self.create_subscription(
            Detection2D, '/detections', self.detection_callback, 10)

        # Action
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.latest_detections = []
        self.target_locked = False

        self.get_logger().info('Perception-Action Node initialized')

    def detection_callback(self, msg):
        """Update latest detections"""
        self.latest_detections = [msg]  # Simplified - in practice, maintain list

    def image_callback(self, msg):
        """Process image and trigger actions if needed"""
        if self.latest_detections and not self.target_locked:
            # Perform action based on detection
            self.approach_target()

    def approach_target(self):
        """Approach detected target"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.2  # Move forward slowly
        cmd_msg.angular.z = 0.0  # Keep straight for now
        self.cmd_vel_pub.publish(cmd_msg)

        self.get_logger().info('Approaching detected target')

def main(args=None):
    rclpy.init(args=args)
    perception_action = PerceptionActionNode()

    try:
        rclpy.spin(perception_action)
    except KeyboardInterrupt:
        # Stop robot before shutting down
        cmd_msg = Twist()
        perception_action.cmd_vel_pub.publish(cmd_msg)
        pass
    finally:
        perception_action.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Acceptance Criteria Met

- [X] Feature detection and object recognition techniques
- [X] OpenCV integration examples
- [X] Point cloud processing basics