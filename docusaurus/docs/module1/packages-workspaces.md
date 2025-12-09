# ROS 2 Packages and Workspaces

## Introduction

ROS 2 organizes code into packages and workspaces. Understanding this structure is crucial for developing and managing ROS 2 applications.

## Workspaces

A workspace is a directory that contains ROS 2 packages. It's the top-level directory where you'll organize your ROS 2 development.

### Creating a Workspace

```bash
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace
```

### Building a Workspace

After adding packages to your workspace, you need to build them using colcon:

```bash
colcon build
```

This will create the following directories:
- `build/` - Build artifacts
- `install/` - Installation directory with executables and libraries
- `log/` - Build logs

## Packages

A package is the basic building unit in ROS 2. It contains source code, configuration files, and other resources needed for a specific functionality.

### Creating a Package

```bash
cd ~/ros2_workspace/src
ros2 pkg create --build-type ament_python my_robot_package
```

For C++ packages:
```bash
ros2 pkg create --build-type ament_cmake my_robot_package
```

### Package Structure

A typical ROS 2 package includes:
- `package.xml` - Package manifest with metadata
- `CMakeLists.txt` - Build configuration for CMake packages
- `setup.py` - Build configuration for Python packages
- `src/` - Source code files
- `include/` - Header files (C++)
- `launch/` - Launch files
- `config/` - Configuration files
- `test/` - Test files

## Colcon Build System

Colcon is the build tool used in ROS 2. It's designed to build multiple packages in a workspace efficiently.

### Common Colcon Commands

- `colcon build` - Build all packages in the workspace
- `colcon build --packages-select <pkg_name>` - Build specific packages
- `colcon build --symlink-install` - Use symlinks for easier development
- `colcon test` - Run tests for all packages
- `colcon test-result --all` - Show test results

## Debugging Techniques

### Common Issues and Solutions

1. **Package not found**: Source your workspace after building:
   ```bash
   source install/setup.bash
   ```

2. **Import errors**: Make sure you've sourced the correct setup file

3. **Build failures**: Check dependencies in `package.xml` and ensure all required packages are available

4. **Workspace overlay**: If you have multiple workspaces, source them in the correct order

### Debugging Commands

- `ros2 pkg list` - List all available packages
- `ros2 pkg executables <pkg_name>` - List executables in a package
- `ament list_packages` - Alternative way to list packages

## Best Practices

- Organize packages by functionality (e.g., perception, navigation, control)
- Use descriptive package names
- Keep packages focused on a single responsibility
- Document dependencies clearly in package.xml
- Use version control for your workspace

## Acceptance Criteria Met

- [X] Step-by-step package creation guide
- [X] Colcon build system explanation
- [X] Debugging techniques documented