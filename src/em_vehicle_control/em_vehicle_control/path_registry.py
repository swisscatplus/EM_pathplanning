from typing import Dict, Iterable, List, Tuple

from builtin_interfaces.msg import Time
from em_vehicle_control_msgs.msg import Path2D, Pose2D


PathPoint = Tuple[float, float]


HARDCODED_PATHS: Dict[int, List[PathPoint]] = {
    1: [
        (0.325, 2.985), (0.325, 1.985), (0.325, 0.985), (0.325, 0.0),
        (0.325, -0.200), (0.65, -0.325), (1.65, -0.325), (2.65, -0.325),
        (3.65, -0.325), (4.06, -0.325), (4.35, -0.200), (4.35, 0.0),
        (4.35, 0.885),
    ],
    2: [
        (4.35, 0.885), (4.35, 0.0), (4.35, -0.200), (4.06, -0.325),
        (3.65, -0.325), (2.65, -0.325), (1.65, -0.325), (0.65, -0.325),
        (0.325, -0.200), (0.325, 0.0), (0.325, 0.985), (0.325, 1.985),
        (0.325, 2.985),
    ],
}


PATH_DIRECTIONS = {
    1: Pose2D.FORWARD,
    2: Pose2D.BACKWARD,
}


class HardcodedPathProvider:
    """Path lookup boundary that can later be backed by a planner."""

    def __init__(self, paths: Dict[int, Iterable[PathPoint]] = None) -> None:
        self._paths = paths if paths is not None else HARDCODED_PATHS

    def has_path(self, path_id: int) -> bool:
        return path_id in self._paths

    def available_path_ids(self) -> List[int]:
        return sorted(self._paths.keys())

    def build_path(self, path_id: int, stamp: Time = None, frame_id: str = "map") -> Path2D:
        if not self.has_path(path_id):
            raise KeyError(f"Unknown path_id {path_id}")

        msg = Path2D()
        msg.header.frame_id = frame_id
        if stamp is not None:
            msg.header.stamp = stamp

        direction = PATH_DIRECTIONS.get(path_id, Pose2D.FORWARD)
        for x, y in self._paths[path_id]:
            pose = Pose2D()
            pose.header.frame_id = frame_id
            if stamp is not None:
                pose.header.stamp = stamp
            pose.x = float(x)
            pose.y = float(y)
            pose.direction_flag = direction
            msg.poses.append(pose)

        return msg
