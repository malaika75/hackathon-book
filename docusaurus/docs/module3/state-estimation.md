# State Estimation

## Overview

State estimation is a fundamental capability in robotics that involves determining the internal state of a robot (position, orientation, velocity, etc.) from noisy sensor measurements and control inputs. This section covers the theoretical foundations of state estimation, practical implementation of filters like Kalman filters and particle filters, and techniques for sensor fusion.

## Filter Theory Explanations

### 1. Kalman Filters

The Kalman filter is an optimal recursive data processing algorithm that estimates the state of a dynamic system from a series of incomplete and noisy measurements. It's particularly effective for linear systems with Gaussian noise.

#### Mathematical Foundation

The Kalman filter operates in two main steps:

1. **Prediction Step**: Predict the state and its uncertainty based on the previous state and control input
2. **Update Step**: Update the prediction with the actual measurement

#### State Prediction
```
x̂(k|k-1) = F(k) * x̂(k-1|k-1) + B(k) * u(k)
P(k|k-1) = F(k) * P(k-1|k-1) * F(k)ᵀ + Q(k)
```

Where:
- `x̂(k|k-1)` is the predicted state estimate
- `F(k)` is the state transition model
- `B(k)` is the control-input model
- `u(k)` is the control vector
- `P(k|k-1)` is the predicted estimate covariance
- `Q(k)` is the process noise covariance

#### State Update
```
K(k) = P(k|k-1) * H(k)ᵀ * [H(k) * P(k|k-1) * H(k)ᵀ + R(k)]⁻¹
x̂(k|k) = x̂(k|k-1) + K(k) * [z(k) - H(k) * x̂(k|k-1)]
P(k|k) = [I - K(k) * H(k)] * P(k|k-1)
```

Where:
- `K(k)` is the Kalman gain
- `H(k)` is the observation model
- `R(k)` is the observation noise covariance
- `z(k)` is the actual measurement
- `x̂(k|k)` is the updated state estimate
- `P(k|k)` is the updated estimate covariance

### 2. Extended Kalman Filter (EKF)

For non-linear systems, the Extended Kalman Filter linearizes the system around the current estimate using Jacobians.

#### State Prediction (Non-linear)
```
x̂(k|k-1) = f(x̂(k-1|k-1), u(k), k)
P(k|k-1) = F(k) * P(k-1|k-1) * F(k)ᵀ + Q(k)
```

Where `F(k)` is the Jacobian of `f` with respect to the state.

#### State Update (Non-linear)
```
ẑ(k) = h(x̂(k|k-1), k)
K(k) = P(k|k-1) * H(k)ᵀ * [H(k) * P(k|k-1) * H(k)ᵀ + R(k)]⁻¹
x̂(k|k) = x̂(k|k-1) + K(k) * [z(k) - ẑ(k)]
P(k|k) = [I - K(k) * H(k)] * P(k|k-1)
```

Where `H(k)` is the Jacobian of `h` with respect to the state.

### 3. Particle Filters

Particle filters represent the probability distribution of the state using a set of random samples (particles) with associated weights. They are particularly useful for non-linear, non-Gaussian systems.

#### Algorithm Steps:
1. **Initialization**: Generate N particles with random states
2. **Prediction**: Propagate each particle through the motion model
3. **Update**: Compute weights based on how well particles match observations
4. **Resampling**: Resample particles based on their weights
5. **Estimation**: Compute state estimate as weighted average of particles

## Practical Implementation Examples

### 1. Kalman Filter Implementation

```python
#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Optional

class KalmanFilter:
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector (position, velocity, etc.)
        self.x = np.zeros((state_dim, 1))

        # State covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance
        self.Q = np.eye(state_dim)

        # Measurement noise covariance
        self.R = np.eye(measurement_dim)

        # State transition matrix
        self.F = np.eye(state_dim)

        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))

        # Control matrix
        self.B = np.zeros((state_dim, state_dim))

        # Identity matrix
        self.I = np.eye(state_dim)

    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prediction step of the Kalman filter

        Args:
            u: Control input vector (optional)

        Returns:
            Predicted state estimate
        """
        if u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.flatten()

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step of the Kalman filter

        Args:
            z: Measurement vector

        Returns:
            Updated state estimate
        """
        # Innovation (measurement residual)
        y = z.reshape(-1, 1) - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x = self.x + K @ y

        # Updated covariance
        self.P = (self.I - K @ self.H) @ self.P

        return self.x.flatten()

    def set_state(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """Set the state and optionally the covariance"""
        self.x = x.reshape(-1, 1)
        if P is not None:
            self.P = P

# Example: 1D position and velocity tracking
class PositionVelocityKalmanFilter(KalmanFilter):
    def __init__(self, dt: float = 1.0):
        super().__init__(state_dim=2, measurement_dim=1)  # [position, velocity]

        # State transition model (constant velocity model)
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])

        # Measurement model (only position is measured)
        self.H = np.array([[1, 0]])

        # Process noise (model uncertainty)
        self.Q = np.array([
            [0.25*dt**4, 0.5*dt**3],
            [0.5*dt**3, dt**2]
        ])

        # Measurement noise
        self.R = np.array([[0.1]])  # Measurement uncertainty

def example_usage():
    # Create a Kalman filter for position-velocity tracking
    kf = PositionVelocityKalmanFilter(dt=0.1)

    # Initial state: position = 0, velocity = 1
    kf.set_state(np.array([0.0, 1.0]))

    # Simulate measurements with noise
    true_positions = []
    measured_positions = []
    estimated_positions = []

    for t in range(100):
        # True position (with some acceleration)
        true_pos = 0.5 * 0.1 * t**2 + 1.0 * t
        true_positions.append(true_pos)

        # Noisy measurement
        noise = np.random.normal(0, 0.5)
        measured_pos = true_pos + noise
        measured_positions.append(measured_pos)

        # Update filter with measurement
        estimated_state = kf.update(np.array([measured_pos]))
        estimated_positions.append(estimated_state[0])

        # Predict for next time step (no control input)
        kf.predict()

    print("Kalman Filter Example Completed")
    print(f"Final estimated position: {estimated_positions[-1]:.2f}")
    print(f"Final true position: {true_positions[-1]:.2f}")

if __name__ == "__main__":
    example_usage()
```

### 2. Particle Filter Implementation

```python
#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, num_particles: int, state_dim: int,
                 motion_model: Callable, measurement_model: Callable,
                 process_noise: np.ndarray, measurement_noise: np.ndarray):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initialize particles
        self.particles = np.random.normal(0, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, u: np.ndarray):
        """Predict step: propagate particles through motion model"""
        for i in range(self.num_particles):
            # Add process noise to motion model
            noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.process_noise)
            self.particles[i] = self.motion_model(self.particles[i], u) + noise

    def update(self, z: np.ndarray):
        """Update step: compute weights based on measurements"""
        for i in range(self.num_particles):
            # Compute expected measurement for this particle
            expected_z = self.measurement_model(self.particles[i])

            # Compute likelihood of actual measurement given particle state
            innovation = z - expected_z
            innovation_cov = self.measurement_noise

            # Compute likelihood (Gaussian)
            likelihood = self.gaussian_pdf(innovation, np.zeros_like(innovation), innovation_cov)

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Avoid numerical issues
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on their weights"""
        # Systematic resampling
        indices = self.systematic_resample()

        # Resample particles and reset weights
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def systematic_resample(self) -> np.ndarray:
        """Systematic resampling algorithm"""
        N = self.num_particles
        indices = np.zeros(N, dtype=int)

        # Generate random starting point
        random_start = np.random.random() / N

        # Compute cumulative weights
        cumulative_weights = np.cumsum(self.weights)

        # Select particles
        i, j = 0, 0
        while i < N:
            while cumulative_weights[j] < random_start + i / N:
                j += 1
            indices[i] = j
            i += 1

        return indices

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute state estimate as weighted average of particles"""
        # Compute weighted mean
        mean = np.average(self.particles, axis=0, weights=self.weights)

        # Compute weighted covariance
        diff = self.particles - mean
        cov = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.num_particles):
            cov += self.weights[i] * np.outer(diff[i], diff[i])

        return mean, cov

    def gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Compute Gaussian probability density function"""
        dim = len(x)
        diff = x - mean

        # Compute normalization constant
        norm_const = 1.0 / np.sqrt((2 * np.pi)**dim * np.linalg.det(cov))

        # Compute exponential term
        exp_term = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)

        return norm_const * exp_term

# Example: 2D position tracking
def motion_model_2d(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Simple motion model: x(k+1) = x(k) + v*dt"""
    dt = 0.1
    new_state = state.copy()
    new_state[0] += state[2] * dt  # Update x position
    new_state[1] += state[3] * dt  # Update y position
    new_state[2] += control[0] * dt  # Update x velocity
    new_state[3] += control[1] * dt  # Update y velocity
    return new_state

def measurement_model_2d(state: np.ndarray) -> np.ndarray:
    """Measurement model: measure only position"""
    return state[:2]  # Return [x, y]

def example_usage():
    # Define process and measurement noise
    process_noise = np.diag([0.1, 0.1, 0.5, 0.5])  # [x_pos, y_pos, x_vel, y_vel]
    measurement_noise = np.diag([0.5, 0.5])  # [x_pos, y_pos]

    # Create particle filter
    pf = ParticleFilter(
        num_particles=1000,
        state_dim=4,  # [x_pos, y_pos, x_vel, y_vel]
        motion_model=motion_model_2d,
        measurement_model=measurement_model_2d,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )

    # Initialize particles around initial estimate
    initial_pos = np.array([0.0, 0.0])
    initial_vel = np.array([1.0, 0.5])
    initial_state = np.concatenate([initial_pos, initial_vel])

    # Add some uncertainty to initial particles
    for i in range(pf.num_particles):
        pf.particles[i] = initial_state + np.random.multivariate_normal(
            np.zeros(4), np.diag([0.5, 0.5, 0.5, 0.5])
        )

    # Simulate tracking
    true_states = []
    estimated_states = []

    for t in range(50):
        # True state (with some acceleration)
        true_pos = np.array([0.1 * t**2 + 1.0 * t, 0.05 * t**2 + 0.5 * t])
        true_vel = np.array([0.2 * t + 1.0, 0.1 * t + 0.5])
        true_state = np.concatenate([true_pos, true_vel])
        true_states.append(true_state)

        # Simulate noisy measurement
        measurement_noise_sample = np.random.multivariate_normal(
            np.zeros(2), measurement_noise
        )
        measurement = true_pos + measurement_noise_sample

        # Predict step (with zero control input)
        control = np.array([0.0, 0.0])
        pf.predict(control)

        # Update step
        pf.update(measurement)

        # Resample if effective sample size is low
        effective_samples = 1.0 / np.sum(pf.weights**2)
        if effective_samples < pf.num_particles / 2:
            pf.resample()

        # Get estimate
        estimate, _ = pf.estimate()
        estimated_states.append(estimate)

    print("Particle Filter Example Completed")
    print(f"Final estimated position: [{estimated_states[-1][0]:.2f}, {estimated_states[-1][1]:.2f}]")
    print(f"Final true position: [{true_states[-1][0]:.2f}, {true_states[-1][1]:.2f}]")

if __name__ == "__main__":
    example_usage()
```

### 3. ROS 2 Integration Example

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import math

class RobotStateEstimator(Node):
    def __init__(self):
        super().__init__('robot_state_estimator')

        # Initialize Kalman filter for robot state estimation
        self.kf = self.initialize_kalman_filter()

        # Subscribers for sensor data
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publisher for estimated state
        self.estimated_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/estimated_pose', 10)

        # TF broadcaster for robot pose
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for prediction step
        self.timer = self.create_timer(0.05, self.prediction_step)  # 20 Hz

        # Store last odometry for velocity calculation
        self.last_odom_time = self.get_clock().now()
        self.last_odom_pose = None

        # State variables
        self.current_state = np.zeros(6)  # [x, y, theta, vx, vy, omega]
        self.is_initialized = False

        self.get_logger().info('Robot State Estimator initialized')

    def initialize_kalman_filter(self):
        """Initialize Kalman filter for robot state estimation"""
        # State vector: [x, y, theta, vx, vy, omega]
        state_dim = 6
        measurement_dim = 3  # [x, y, theta] from odometry

        kf = KalmanFilter(state_dim=state_dim, measurement_dim=measurement_dim)

        # Set initial covariance (high uncertainty)
        kf.P = np.diag([1.0, 1.0, 0.1, 1.0, 1.0, 0.1])  # [x, y, theta, vx, vy, omega]

        # Process noise (motion model uncertainty)
        kf.Q = np.diag([0.1, 0.1, 0.05, 0.5, 0.5, 0.1])  # [x, y, theta, vx, vy, omega]

        # Measurement noise (sensor uncertainty)
        kf.R = np.diag([0.05, 0.05, 0.02])  # [x, y, theta]

        # Measurement matrix (we only measure position and orientation)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x position
            [0, 1, 0, 0, 0, 0],  # y position
            [0, 0, 1, 0, 0, 0]   # orientation
        ])

        return kf

    def prediction_step(self):
        """Prediction step using control input (if available)"""
        if not self.is_initialized:
            return

        # Get current time
        current_time = self.get_clock().now()
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9

        if dt > 0:
            # Update state transition matrix based on time step
            self.kf.F = np.array([
                [1, 0, 0, dt, 0,  0],     # x = x + vx*dt
                [0, 1, 0, 0,  dt, 0],     # y = y + vy*dt
                [0, 0, 1, 0,  0,  dt],    # theta = theta + omega*dt
                [0, 0, 0, 1,  0,  0],     # vx = vx (constant velocity model)
                [0, 0, 0, 0,  1,  0],     # vy = vy
                [0, 0, 0, 0,  0,  1]      # omega = omega
            ])

            # Predict state (no control input in this example)
            predicted_state = self.kf.predict()
            self.current_state = predicted_state

    def odom_callback(self, msg: Odometry):
        """Handle odometry measurements"""
        # Extract pose from odometry message
        pose = msg.pose.pose
        position = pose.position
        orientation = pose.orientation

        # Convert quaternion to Euler angle (yaw)
        yaw = self.quaternion_to_yaw(orientation)

        # Create measurement vector [x, y, theta]
        measurement = np.array([position.x, position.y, yaw])

        if not self.is_initialized:
            # Initialize state with first measurement
            initial_state = np.zeros(6)
            initial_state[0] = position.x  # x
            initial_state[1] = position.y  # y
            initial_state[2] = yaw         # theta
            self.kf.set_state(initial_state)
            self.current_state = initial_state
            self.is_initialized = True
            self.last_odom_pose = [position.x, position.y, yaw]
            self.last_odom_time = self.get_clock().now()
            return

        # Update Kalman filter with measurement
        self.kf.update(measurement)

        # Update current state
        self.current_state = self.kf.x.flatten()

        # Calculate velocity from odometry if possible
        current_time = self.get_clock().now()
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9

        if dt > 0 and self.last_odom_pose is not None:
            dx = position.x - self.last_odom_pose[0]
            dy = position.y - self.last_odom_pose[1]
            dtheta = yaw - self.last_odom_pose[2]

            # Normalize angle difference
            dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

            # Update velocity estimates in state
            self.current_state[3] = dx / dt  # vx
            self.current_state[4] = dy / dt  # vy
            self.current_state[5] = dtheta / dt  # omega

            # Update state covariance for velocity estimates
            self.kf.P[3, 3] = 0.1  # Lower uncertainty for velocity
            self.kf.P[4, 4] = 0.1
            self.kf.P[5, 5] = 0.05

        # Update last values
        self.last_odom_pose = [position.x, position.y, yaw]
        self.last_odom_time = current_time

        # Publish estimated pose
        self.publish_estimated_pose()

    def imu_callback(self, msg: Imu):
        """Handle IMU measurements for orientation and angular velocity"""
        if not self.is_initialized:
            return

        # Extract orientation from IMU
        orientation = msg.orientation
        yaw = self.quaternion_to_yaw(orientation)

        # Extract angular velocity
        angular_velocity = msg.angular_velocity.z

        # Update orientation in state
        self.current_state[2] = yaw
        self.current_state[5] = angular_velocity

        # Also update the Kalman filter state
        self.kf.x[2] = yaw
        self.kf.x[5] = angular_velocity

    def scan_callback(self, msg: LaserScan):
        """Handle laser scan data for position updates"""
        if not self.is_initialized:
            return

        # This is a simplified example - in practice, you would use
        # scan matching or landmark detection to get position measurements
        # For now, we'll just use the scan to validate our position estimate
        pass

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def publish_estimated_pose(self):
        """Publish the estimated robot pose"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Set pose
        pose_msg.pose.pose.position.x = float(self.current_state[0])
        pose_msg.pose.pose.position.y = float(self.current_state[1])
        pose_msg.pose.pose.position.z = 0.0

        # Convert orientation from yaw to quaternion
        yaw = self.current_state[2]
        pose_msg.pose.pose.orientation.z = math.sin(yaw / 2)
        pose_msg.pose.pose.orientation.w = math.cos(yaw / 2)

        # Set covariance
        pose_msg.pose.covariance[0] = float(self.kf.P[0, 0])  # x
        pose_msg.pose.covariance[7] = float(self.kf.P[1, 1])  # y
        pose_msg.pose.covariance[35] = float(self.kf.P[2, 2])  # yaw

        self.estimated_pose_pub.publish(pose_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(self.current_state[0])
        t.transform.translation.y = float(self.current_state[1])
        t.transform.translation.z = 0.0

        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStateEstimator()

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

## Sensor Fusion Techniques

### 1. Multi-Sensor Data Integration

Sensor fusion combines data from multiple sensors to achieve better accuracy and reliability than could be achieved by using a single sensor. Common sensor combinations include:

- **IMU + Odometry**: Combines inertial measurements with wheel encoders
- **GPS + IMU**: Combines global positioning with local motion sensing
- **Camera + LiDAR**: Combines visual and range data
- **Wheel Encoders + Visual Odometry**: Combines proprioceptive and exteroceptive sensing

### 2. Information Form Filters

Instead of working with covariance matrices, information form filters work with information matrices (inverse of covariance). This can be more numerically stable and efficient for certain applications.

```python
class InformationFilter:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.y = np.zeros((state_dim, 1))  # Information state vector
        self.Y = np.zeros((state_dim, state_dim))  # Information matrix

    def prediction(self, F: np.ndarray, Q: np.ndarray):
        """Prediction step in information form"""
        # Convert to covariance form
        P = np.linalg.inv(self.Y)
        x = np.linalg.inv(self.Y) @ self.y

        # Standard Kalman prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Convert back to information form
        self.Y = np.linalg.inv(P_pred)
        self.y = self.Y @ x_pred

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Update step in information form"""
        # Information contribution from measurement
        Y_meas = H.T @ np.linalg.inv(R) @ H
        y_meas = H.T @ np.linalg.inv(R) @ z.reshape(-1, 1)

        # Combine with prior information
        self.Y = self.Y + Y_meas
        self.y = self.y + y_meas
```

### 3. Covariance Intersection

For combining estimates from independent sources when the correlation between them is unknown:

```python
def covariance_intersection(est1: np.ndarray, cov1: np.ndarray,
                          est2: np.ndarray, cov2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine two estimates using covariance intersection
    """
    # Compute the combination parameter (omega)
    S = np.linalg.inv(cov1) + np.linalg.inv(cov2)
    omega = np.linalg.inv(cov1) @ np.linalg.inv(S)

    # Combined covariance
    P_combined = np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2))

    # Combined estimate
    x_combined = P_combined @ (np.linalg.inv(cov1) @ est1 + np.linalg.inv(cov2) @ est2)

    return x_combined, P_combined
```

## Best Practices for State Estimation

### 1. Model Accuracy
- Use appropriate motion models for your application
- Properly tune process and measurement noise parameters
- Validate models against real-world data
- Consider environmental factors in noise modeling

### 2. Numerical Stability
- Use square-root filters to maintain positive definiteness
- Implement checks for numerical errors
- Use appropriate data types (double precision for covariance)
- Monitor condition numbers of matrices

### 3. Computational Efficiency
- Use sparse matrix techniques when appropriate
- Implement efficient resampling for particle filters
- Consider reduced-order filters when possible
- Optimize for real-time performance requirements

### 4. Validation and Testing
- Test with various noise levels
- Validate against ground truth when available
- Monitor filter consistency (NEES - Normalized Estimation Error Squared)
- Implement failure detection mechanisms

## Acceptance Criteria Met

- [X] Filter theory explanations
- [X] Practical implementation examples
- [X] Sensor fusion techniques