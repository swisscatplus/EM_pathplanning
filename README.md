<br />
<div align="center">
  <a href="https://github.com/swisscatplus/EM_pathplanning">
    <img src="pictures/logo.png" alt="Logo" height="80">
  </a>
  <h1 align="center">EM_pathplanning</h1>
</div>

ROS 2 laptop-side path planning and path tracking stack for EM Robot.

This repository is responsible for:

- on-robot path planning logic
- path tracking and control nodes
- custom path and pose messages/services
- Dockerized laptop development on ROS 2 Humble

Fleet-level multi-robot orchestration lives in `EM_fleetmanager`; robot-side hardware runtime lives in `EM_onrobot`.

## Supported Target

The supported runtime target is an Ubuntu laptop/workstation running ROS 2 Humble in Docker.

The default container command is:

```bash
ros2 launch em_vehicle_control tracker_loop.launch.py
```

## Docker Development

Start the laptop development container:

```bash
./scripts/deploy_dev.sh
```

Run in the foreground instead:

```bash
./scripts/start_dev.sh
```

Useful follow-up commands:

```bash
./scripts/start_dev.sh logs
./scripts/start_dev.sh shell
./scripts/start_dev.sh down
```

The compose workflow bind-mounts this repository into `/ros2_ws`, builds `em_vehicle_control_msgs` and `em_vehicle_control` with `--symlink-install`, sources the workspace, then launches `tracker_loop.launch.py`.

## Image Build

Build the laptop image:

```bash
./scripts/deployImage.sh
```

Override the image name or push it:

```bash
IMAGE_NAME=ghcr.io/swisscatplus/em_pathplanning:latest ./scripts/deployImage.sh
PUSH_IMAGE=1 IMAGE_NAME=ghcr.io/swisscatplus/em_pathplanning:latest ./scripts/deployImage.sh
```

## Build And Test

Build the ROS workspace locally:

```bash
colcon build --symlink-install --packages-select em_vehicle_control_msgs em_vehicle_control
```

Run tests:

```bash
colcon test --packages-select em_vehicle_control_msgs em_vehicle_control
colcon test-result --verbose
```

Generated `build/`, `install/`, and `log/` directories are ignored by Git.

## Repository Layout

```text
EM_pathplanning/
├── config/
│   └── fastdds.xml
├── docker/
│   ├── compose.yaml
│   ├── Dockerfile
│   └── entrypoint.sh
├── scripts/
│   ├── deploy_dev.sh
│   ├── deployImage.sh
│   └── start_dev.sh
├── src/
│   ├── em_vehicle_control/
│   └── em_vehicle_control_msgs/
└── pictures/
```

The original exploratory scripts that were outside ROS packages now live under `em_vehicle_control.examples`.

See `JasperTanMsCSlides.pdf` for design background.
