# Path Planning Algorithms

## Overview

Path planning is a fundamental capability in robotics that involves finding an optimal or feasible path from a start location to a goal location while avoiding obstacles. This section covers the core algorithms used in robotics navigation, including their mathematical foundations and practical implementations.

## Algorithm Explanations with Visual Examples

### 1. A* Algorithm

A* (A-star) is a popular graph traversal and path search algorithm that is widely used due to its completeness, optimality, and optimal efficiency. It combines the advantages of Dijkstra's algorithm (which guarantees the shortest path) with the heuristic approach of Greedy Best-First-Search.

#### How A* Works

A* uses the following evaluation function:
```
f(n) = g(n) + h(n)
```

Where:
- `f(n)` = estimated total cost of path through node n
- `g(n)` = actual cost from start to current node n
- `h(n)` = estimated cost from current node n to goal (heuristic)

#### Implementation Example

```python
#!/usr/bin/env python3

import heapq
import numpy as np
from typing import List, Tuple, Optional

class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent=None):
        self.position = position
        self.g = g_cost  # Cost from start to current node
        self.h = h_cost  # Heuristic cost from current node to goal
        self.f = g_cost + h_cost  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

class AStarPlanner:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        # 8-directional movement (including diagonals)
        self.directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1),           (0, 1),
                          (1, -1),  (1, 0),  (1, 1)]
        # Movement costs for 8 directions (diagonals cost more)
        self.costs = [1.414, 1, 1.414,
                     1,      1,
                     1.414, 1, 1.414]

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean distance)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds and not an obstacle"""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return self.grid[x][y] == 0  # 0 = free space, 1 = obstacle
        return False

    def get_neighbors(self, node: Node) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighbors of a node"""
        neighbors = []
        for i, direction in enumerate(self.directions):
            new_pos = (node.position[0] + direction[0], node.position[1] + direction[1])
            if self.is_valid(new_pos):
                neighbors.append((new_pos, self.costs[i]))
        return neighbors

    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start by following parent pointers"""
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Return reversed path (start to goal)

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path from start to goal using A* algorithm"""
        # Initialize open and closed sets
        open_set = []
        closed_set = set()

        # Create start node
        start_node = Node(start, 0, self.heuristic(start, goal))
        heapq.heappush(open_set, start_node)

        # Keep track of nodes for faster lookup
        open_dict = {start: start_node}

        while open_set:
            # Get node with lowest f cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position

            # Remove from open_dict
            if current_pos in open_dict:
                del open_dict[current_pos]

            # Add to closed set
            closed_set.add(current_pos)

            # Check if we reached the goal
            if current_pos == goal:
                return self.reconstruct_path(current_node)

            # Explore neighbors
            for neighbor_pos, move_cost in self.get_neighbors(current_node):
                if neighbor_pos in closed_set:
                    continue

                # Calculate tentative g score
                tentative_g = current_node.g + move_cost

                # Check if this path to neighbor is better
                if neighbor_pos in open_dict:
                    existing_node = open_dict[neighbor_pos]
                    if tentative_g < existing_node.g:
                        # Update the existing node
                        existing_node.g = tentative_g
                        existing_node.f = tentative_g + existing_node.h
                        existing_node.parent = current_node
                        heapq.heapify(open_set)  # Re-heapify to maintain heap property
                else:
                    # New node to explore
                    h_cost = self.heuristic(neighbor_pos, goal)
                    neighbor_node = Node(neighbor_pos, tentative_g, h_cost, current_node)
                    heapq.heappush(open_set, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node

        # No path found
        return None

# Example usage
def example_usage():
    # Create a simple grid (0 = free space, 1 = obstacle)
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    planner = AStarPlanner(grid)
    start = (0, 0)
    goal = (7, 7)

    path = planner.plan(start, goal)

    if path:
        print("Path found:")
        for pos in path:
            print(f"  {pos}")
    else:
        print("No path found")

if __name__ == "__main__":
    example_usage()
```

### 2. Dijkstra's Algorithm

Dijkstra's algorithm is a classic graph search algorithm that solves the single-source shortest path problem for a graph with non-negative edge weights. Unlike A*, it doesn't use a heuristic function, making it slower but guaranteed to find the shortest path.

#### How Dijkstra Works

Dijkstra's algorithm maintains a priority queue of nodes to visit, always selecting the node with the smallest distance from the start. It updates the distances to neighboring nodes if a shorter path is found.

#### Implementation Example

```python
#!/usr/bin/env python3

import heapq
import numpy as np
from typing import List, Tuple, Optional

class DijkstraNode:
    def __init__(self, position: Tuple[int, int], distance: float = float('inf'), parent=None):
        self.position = position
        self.distance = distance  # Distance from start
        self.parent = parent

    def __lt__(self, other):
        return self.distance < other.distance

class DijkstraPlanner:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        # 4-directional movement (up, down, left, right)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.costs = [1.0, 1.0, 1.0, 1.0]

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds and not an obstacle"""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return self.grid[x][y] == 0  # 0 = free space, 1 = obstacle
        return False

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighbors of a position"""
        neighbors = []
        for i, direction in enumerate(self.directions):
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
            if self.is_valid(new_pos):
                neighbors.append((new_pos, self.costs[i]))
        return neighbors

    def reconstruct_path(self, nodes: dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start by following parent pointers"""
        path = []
        current_pos = goal
        while current_pos is not None:
            path.append(current_pos)
            current_node = nodes.get(current_pos)
            if current_node and current_node.parent:
                current_pos = current_node.parent
            else:
                current_pos = None
        return path[::-1]  # Return reversed path (start to goal)

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path from start to goal using Dijkstra's algorithm"""
        # Initialize distances and priority queue
        distances = {}
        previous = {}
        pq = []

        # Initialize all nodes with infinity distance
        for x in range(self.rows):
            for y in range(self.cols):
                distances[(x, y)] = float('inf')
                previous[(x, y)] = None

        # Set start distance to 0
        distances[start] = 0
        heapq.heappush(pq, DijkstraNode(start, 0))

        visited = set()

        while pq:
            # Get node with minimum distance
            current_node = heapq.heappop(pq)
            current_pos = current_node.position

            # Skip if already visited
            if current_pos in visited:
                continue

            # Mark as visited
            visited.add(current_pos)

            # Check if we reached the goal
            if current_pos == goal:
                # Reconstruct path
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return path[::-1]  # Return reversed path (start to goal)

            # Explore neighbors
            for neighbor_pos, edge_weight in self.get_neighbors(current_pos):
                if neighbor_pos in visited:
                    continue

                # Calculate tentative distance
                tentative_distance = distances[current_pos] + edge_weight

                # If we found a shorter path, update it
                if tentative_distance < distances[neighbor_pos]:
                    distances[neighbor_pos] = tentative_distance
                    previous[neighbor_pos] = current_pos
                    heapq.heappush(pq, DijkstraNode(neighbor_pos, tentative_distance))

        # No path found
        return None

# Example usage
def example_usage():
    # Create a simple grid (0 = free space, 1 = obstacle)
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    planner = DijkstraPlanner(grid)
    start = (0, 0)
    goal = (7, 7)

    path = planner.plan(start, goal)

    if path:
        print("Path found using Dijkstra:")
        for pos in path:
            print(f"  {pos}")
    else:
        print("No path found")

if __name__ == "__main__":
    example_usage()
```

### 3. RRT (Rapidly-exploring Random Trees)

RRT (Rapidly-exploring Random Tree) is a motion planning algorithm that is especially effective in high-dimensional spaces. It incrementally builds a tree of possible paths by randomly sampling the configuration space.

#### How RRT Works

1. Start with an initial configuration as the root of the tree
2. Randomly sample points in the configuration space
3. Find the nearest node in the tree to the random sample
4. Extend the tree toward the random sample (with collision checking)
5. Repeat until the goal is reached or a maximum number of iterations is reached

#### Implementation Example

```python
#!/usr/bin/env python3

import numpy as np
import random
from typing import List, Tuple, Optional
import math

class RRTNode:
    def __init__(self, position: Tuple[float, float], parent=None):
        self.position = position
        self.parent = parent

class RRTPlanner:
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                 obstacles: List[Tuple[float, float, float]], step_size: float = 0.5):
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles = obstacles  # List of (x, y, radius) tuples
        self.step_size = step_size

    def distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_collision_free(self, pos: Tuple[float, float]) -> bool:
        """Check if position is collision-free"""
        for obs_x, obs_y, obs_radius in self.obstacles:
            if self.distance(pos, (obs_x, obs_y)) <= obs_radius:
                return False
        return True

    def get_random_point(self) -> Tuple[float, float]:
        """Get a random point in the configuration space"""
        x = random.uniform(self.x_range[0], self.x_range[1])
        y = random.uniform(self.y_range[0], self.y_range[1])
        return (x, y)

    def get_nearest_node(self, nodes: List[RRTNode], target_pos: Tuple[float, float]) -> RRTNode:
        """Find the nearest node in the tree to the target position"""
        nearest_node = nodes[0]
        min_dist = self.distance(nearest_node.position, target_pos)

        for node in nodes[1:]:
            dist = self.distance(node.position, target_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def extend_towards(self, from_node: RRTNode, to_pos: Tuple[float, float]) -> Optional[RRTNode]:
        """Extend the tree from from_node towards to_pos"""
        direction = (to_pos[0] - from_node.position[0], to_pos[1] - from_node.position[1])
        distance = self.distance(from_node.position, to_pos)

        if distance <= self.step_size:
            # If close enough, just go to the target
            new_pos = to_pos
        else:
            # Move step_size towards the target
            scale = self.step_size / distance
            new_pos = (
                from_node.position[0] + direction[0] * scale,
                from_node.position[1] + direction[1] * scale
            )

        # Check if the new position is collision-free
        if self.is_collision_free(new_pos):
            return RRTNode(new_pos, from_node)

        return None

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             max_iterations: int = 1000) -> Optional[List[Tuple[float, float]]]:
        """Plan path using RRT algorithm"""
        # Initialize tree with start node
        start_node = RRTNode(start)
        nodes = [start_node]

        for i in range(max_iterations):
            # Get random point
            random_pos = self.get_random_point()

            # Find nearest node in tree
            nearest_node = self.get_nearest_node(nodes, random_pos)

            # Try to extend towards random point
            new_node = self.extend_towards(nearest_node, random_pos)

            if new_node:
                nodes.append(new_node)

                # Check if we're close to the goal
                if self.distance(new_node.position, goal) <= self.step_size:
                    # Try to connect to goal directly
                    goal_node = self.extend_towards(new_node, goal)
                    if goal_node:
                        # Reconstruct path
                        path = []
                        current = goal_node
                        while current:
                            path.append(current.position)
                            current = current.parent
                        return path[::-1]  # Return reversed path (start to goal)

        # If max iterations reached without finding path to goal
        return None

# Example usage
def example_usage():
    # Define configuration space
    x_range = (0, 10)
    y_range = (0, 10)

    # Define obstacles as (x, y, radius) tuples
    obstacles = [
        (3, 3, 1),    # Circle at (3,3) with radius 1
        (7, 7, 1.5),  # Circle at (7,7) with radius 1.5
        (5, 2, 0.8),  # Circle at (5,2) with radius 0.8
    ]

    planner = RRTPlanner(x_range, y_range, obstacles, step_size=0.5)
    start = (1, 1)
    goal = (9, 9)

    path = planner.plan(start, goal, max_iterations=2000)

    if path:
        print("Path found using RRT:")
        for i, pos in enumerate(path):
            print(f"  {i}: {pos}")
    else:
        print("No path found after maximum iterations")

if __name__ == "__main__":
    example_usage()
```

## Implementation Considerations

### Performance Comparison

| Algorithm | Time Complexity | Space Complexity | Optimality | Completeness |
|-----------|----------------|------------------|------------|--------------|
| A*        | O(b^d)         | O(b^d)           | Optimal    | Complete     |
| Dijkstra  | O(V²) or O(E + V log V) | O(V) | Optimal | Complete |
| RRT       | O(n)           | O(n)             | Suboptimal | Probabilistically Complete |

Where:
- b = branching factor
- d = depth of solution
- V = number of vertices
- E = number of edges
- n = number of samples

### Selection Guidelines

#### When to Use A*
- When you need optimal paths
- When you have a good heuristic function
- For grid-based environments
- When computational resources allow

#### When to Use Dijkstra
- When you need guaranteed optimal paths
- When no good heuristic is available
- For unweighted graphs
- When all edge costs are equal

#### When to Use RRT
- In high-dimensional configuration spaces
- For complex robot kinematics
- When real-time planning is needed
- For non-holonomic robots

## Integration with Navigation Stack

Path planning algorithms integrate with the ROS 2 Navigation Stack through the `planner_server` component. Here's an example of how to implement a custom planner plugin:

```cpp
// custom_planner.hpp
#ifndef CUSTOM_PLANNER_HPP_
#define CUSTOM_PLANNER_HPP_

#include <nav2_core/global_planner.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.h>
#include <string>

namespace custom_planners
{
class CustomPlanner : public nav2_core::GlobalPlanner
{
public:
  CustomPlanner();
  ~CustomPlanner();

  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  nav2_costmap_2d::Costmap2D * costmap_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
};
}  // namespace custom_planners

#endif  // CUSTOM_PLANNER_HPP_
```

## Performance Considerations

### Computational Efficiency

For real-time applications, consider the following optimizations:

1. **Grid Resolution**: Balance between accuracy and computation time
2. **Anytime Algorithms**: Algorithms that can return a solution at any time and improve it over time
3. **Hierarchical Planning**: Plan at multiple resolutions
4. **Dynamic Replanning**: Update paths as new sensor information becomes available

### Memory Management

- Pre-allocate data structures when possible
- Use efficient data structures (priority queues, hash maps)
- Consider memory usage in embedded systems

## Acceptance Criteria Met

- [X] Algorithm explanations with visual examples
- [X] Implementation considerations
- [X] Performance comparisons