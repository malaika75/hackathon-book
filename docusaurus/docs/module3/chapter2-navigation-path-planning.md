# Chapter 2: Navigation, Path Planning & State Estimation

This chapter covers the ROS 2 Navigation Stack, path planning algorithms, and state estimation techniques.

---

## Navigation Stack Configuration

### Overview

The ROS 2 Navigation Stack (Nav2) provides a complete navigation solution:
- Global and local planners
- Costmap management
- Localization (AMCL)
- Behavior trees for complex behaviors
- Recovery behaviors

### Navigation Launch File

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    map_file = LaunchConfiguration('map')

    # Lifecycle Manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': use_sim_time,
            'node_names': ['map_server', 'planner_server', 'controller_server']
        }]
    )

    # Map Server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # Planner Server
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    # Controller Server
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('params_file'),
        lifecycle_manager, map_server, planner_server, controller_server
    ])
```

### Navigation Parameters

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: nav2_controller::DwbController
      min_vel_x: 0.0
      max_vel_x: 0.5
      min_vel_theta: -0.5
      max_vel_theta: 0.5
      acc_lim_x: 1.0
      acc_lim_theta: 1.0

planner_server:
  ros__parameters:
    planner_plugin: nav2_planner::PlannerC
    expected_planner_frequency: 1.0
```

---

## Path Planning Algorithms

### A* Algorithm

A* combines Dijkstra's algorithm with heuristic approach:

```
f(n) = g(n) + h(n)
```

Where:
- `g(n)` = actual cost from start to node n
- `h(n)` = heuristic cost from node n to goal
- `f(n)` = estimated total cost

```python
import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] == 0

    def plan(self, start, goal):
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if not self.is_valid(neighbor) or neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
```

### Other Path Planning Algorithms

1. **Dijkstra's Algorithm**: Finds shortest path without heuristics
2. **RRT* (Rapidly-exploring Random Trees Star)**: For continuous spaces
3. **DWA (Dynamic Window Approach)**: For local robot control

---

## State Estimation

### Kalman Filters

The Kalman filter is an optimal recursive data processing algorithm for linear systems with Gaussian noise.

#### Two Main Steps:

1. **Prediction Step**: Predict state based on previous state and control input
2. **Update Step**: Update prediction with actual measurement

```python
import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.x = np.zeros((state_dim, 1))  # State
        self.P = np.eye(state_dim)          # Covariance
        self.Q = np.eye(state_dim)          # Process noise
        self.R = np.eye(measurement_dim)   # Measurement noise
        self.F = np.eye(state_dim)          # State transition
        self.H = np.zeros((measurement_dim, state_dim))  # Measurement matrix

    def predict(self, u=None):
        if u is not None:
            self.x = self.F @ self.x + u
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.flatten()

    def update(self, z):
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        return self.x.flatten()
```

### Extended Kalman Filter (EKF)

For non-linear systems, EKF linearizes using Jacobians.

### Particle Filters

Particle filters use random samples to represent probability distribution:

```python
class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, motion_model, u):
        for i in range(self.num_particles):
            self.particles[i] = motion_model(self.particles[i], u)

    def update(self, measurement, measurement_model):
        for i in range(self.num_particles):
            expected = measurement_model(self.particles[i])
            likelihood = np.exp(-0.5 * np.sum((measurement - expected)**2))
            self.weights[i] *= likelihood
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)
```

### Sensor Fusion

Combine data from multiple sensors:

- **IMU + Odometry**: Combine inertial measurements with wheel encoders
- **GPS + IMU**: Global positioning with local motion sensing
- **Camera + LiDAR**: Visual and range data

## Acceptance Criteria Met

- [X] Navigation Stack configuration
- [X] Path planning algorithms explained
- [X] State estimation techniques with implementations
- [X] Sensor fusion examples
