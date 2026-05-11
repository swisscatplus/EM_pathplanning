from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from em_vehicle_control_msgs.msg import Path2D, Pose2D


PathPoint = Tuple[float, float]

DEFAULT_PATHS_FILE = "paths.yaml"
DEFAULT_WORKFLOWS_FILE = "path_workflows.yaml"


def resolve_config_path(config_path: str, default_file: str) -> str:
    if config_path:
        if os.path.isabs(config_path):
            return config_path
        if os.path.exists(config_path):
            return os.path.abspath(config_path)
        pkg = get_package_share_directory("em_vehicle_control")
        return os.path.join(pkg, "config", config_path)

    pkg = get_package_share_directory("em_vehicle_control")
    return os.path.join(pkg, "config", default_file)


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def direction_to_msg_value(direction) -> int:
    if isinstance(direction, str):
        normalized = direction.strip().lower()
        if normalized in ("forward", "forwards", "fwd", "1", "+1"):
            return Pose2D.FORWARD
        if normalized in ("backward", "backwards", "reverse", "rev", "-1"):
            return Pose2D.BACKWARD
        raise ValueError(f"Unknown path direction '{direction}'")

    return Pose2D.FORWARD if int(direction) == 1 else Pose2D.BACKWARD


def msg_value_to_direction(value: int) -> str:
    return "forward" if int(value) == Pose2D.FORWARD else "backward"


def densify_path(points: Iterable[PathPoint], ds: float) -> List[PathPoint]:
    points = list(points)
    if not points or ds <= 0.0:
        return points

    dense = [points[0]]
    for p0, p1 in zip(points[:-1], points[1:]):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue

        n = int(dist / ds)
        for i in range(1, n + 1):
            t = i / (n + 1)
            dense.append((p0[0] + t * dx, p0[1] + t * dy))
        dense.append(p1)

    return dense


class YamlPathProvider:
    """Path lookup backed by config/paths.yaml."""

    def __init__(self, config_path: str = "") -> None:
        self.config_path = resolve_config_path(config_path, DEFAULT_PATHS_FILE)
        config = _load_yaml(self.config_path)

        self.frame_id = str(config.get("frame_id", "map"))
        self.default_direction = direction_to_msg_value(config.get("default_direction", "forward"))
        self.default_densify_ds = float(config.get("default_densify_ds", 0.0) or 0.0)

        self._paths: Dict[int, List[PathPoint]] = {}
        self._directions: Dict[int, int] = {}
        for raw_id, raw_path in (config.get("paths") or {}).items():
            path_id = int(raw_id)
            self._paths[path_id] = self._parse_points(raw_path, path_id)
            self._directions[path_id] = direction_to_msg_value(
                raw_path.get("direction", self.default_direction)
            )

        if not self._paths:
            raise RuntimeError(f"No paths defined in {self.config_path}")

    @staticmethod
    def _parse_points(raw_path: Mapping, path_id: int) -> List[PathPoint]:
        raw_points = raw_path.get("points", [])
        points = []
        for index, point in enumerate(raw_points):
            try:
                x, y = point
                points.append((float(x), float(y)))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid points[{index}] for path {path_id}: {point}"
                ) from exc
        if not points:
            raise RuntimeError(f"Path {path_id} has no points")
        return points

    def has_path(self, path_id: int) -> bool:
        return int(path_id) in self._paths

    def available_path_ids(self) -> List[int]:
        return sorted(self._paths.keys())

    def direction_for_path(self, path_id: int, direction_override=None) -> int:
        if direction_override is not None:
            return direction_to_msg_value(direction_override)
        return self._directions.get(int(path_id), self.default_direction)

    def points_for_path(self, path_id: int, densify_ds: Optional[float] = None) -> List[PathPoint]:
        path_id = int(path_id)
        if not self.has_path(path_id):
            raise KeyError(f"Unknown path_id {path_id}")

        points = self._paths[path_id]
        spacing = self.default_densify_ds if densify_ds is None else float(densify_ds or 0.0)
        return densify_path(points, spacing) if spacing > 0.0 else list(points)

    def build_path(
        self,
        path_id: int,
        stamp: Time = None,
        frame_id: str = None,
        direction_override=None,
        densify_ds: Optional[float] = None,
    ) -> Path2D:
        msg = Path2D()
        msg.header.frame_id = frame_id or self.frame_id
        if stamp is not None:
            msg.header.stamp = stamp

        direction = self.direction_for_path(path_id, direction_override)
        for x, y in self.points_for_path(path_id, densify_ds=densify_ds):
            pose = Pose2D()
            pose.header.frame_id = msg.header.frame_id
            if stamp is not None:
                pose.header.stamp = stamp
            pose.x = float(x)
            pose.y = float(y)
            pose.direction_flag = direction
            msg.poses.append(pose)

        return msg


class PathWorkflow:
    def __init__(self, config_path: str = "", workflow_name: str = "") -> None:
        self.config_path = resolve_config_path(config_path, DEFAULT_WORKFLOWS_FILE)
        config = _load_yaml(self.config_path)

        workflows = config.get("workflows") or {}
        selected = workflow_name or config.get("default_workflow", "")
        if not selected:
            selected = next(iter(workflows), "")
        if selected not in workflows:
            raise RuntimeError(f"Workflow '{selected}' not found in {self.config_path}")

        raw_workflow = workflows[selected] or {}
        self.name = selected
        self.manual_mode = bool(raw_workflow.get("manual_mode", config.get("manual_mode", False)))
        self.timer_period_s = float(raw_workflow.get("timer_period_s", config.get("timer_period_s", 80.0)))
        self.densify_ds = float(raw_workflow.get("densify_ds", config.get("densify_ds", 0.0)) or 0.0)
        self.start_immediately = bool(
            raw_workflow.get("start_immediately", config.get("start_immediately", True))
        )
        self.steps = self._parse_steps(raw_workflow.get("steps", []))
        if not self.steps:
            raise RuntimeError(f"Workflow '{selected}' has no steps")

    @staticmethod
    def _parse_steps(raw_steps) -> List[dict]:
        steps = []
        for index, raw_step in enumerate(raw_steps):
            if isinstance(raw_step, int):
                steps.append({"path_id": int(raw_step), "direction": None})
                continue
            try:
                steps.append({
                    "path_id": int(raw_step["path_id"]),
                    "direction": raw_step.get("direction"),
                })
            except Exception as exc:
                raise RuntimeError(f"Invalid workflow step {index}: {raw_step}") from exc
        return steps


# Compatibility name for existing imports.
HardcodedPathProvider = YamlPathProvider
