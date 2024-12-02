"""
Helper functions for BT
To simulate ROS behaviour, without ROS
"""

from typing import List, Tuple
import random
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from em_vehicle_control.helper_classes.map import RoadMap

PosePt2D = Tuple[float, float, float]
PosPt2D = Tuple[float, float]

ROAD_POINTS = [
    (3.24, 1.92),
    (3.11, 5.17),
    (3.15, 8.28),
    (3.24, 13.17),
    (4.98, 5.26),
    (6.26, 1.74),
    (6.03, 8.46),
    (7.31, 0.55),
    (7.54, 4.85),
    (7.39, 11.85),
    (7.22, 15.45),
]


def sim_get_poses_and_goals(N: int = 1) -> List[Tuple[str, PosePt2D, PosePt2D]]:
    """
    Args:
        N(int): Number of robots to simulate (Max 11)

    Returns:
        List[Tuple[str, PosePt2D, PosePt2D]]: List of Tuples of:
            - robot_name(str): Robot name
            - pose(PosePt2D): Current pose of robot
            - goal(PosePt2D): Final goal of robot
    """
    if N > 11:
        N = 11
        print("Reject N > 11 for this helper function, setting N = 11")
    dirs = [0, np.pi / 4, np.pi / 2, np.pi / 4 * 3]
    possible_start_poses = ROAD_POINTS.copy()
    possible_end_poses = ROAD_POINTS.copy()
    random.shuffle(possible_start_poses)
    random.shuffle(possible_end_poses)
    poses_and_goals = []

    for n in range(N):
        # randomly pick a point from possible_start_poses, then pop it
        # randomly pick a point from possible_end_poses, ensure not the same as start, then pop it
        # randomly pick a direction from dir for each start and end poses
        start_pose = possible_start_poses.pop()
        end_pose = random.choice(
            [pose for pose in possible_end_poses if pose != start_pose]
        )
        possible_end_poses.remove(end_pose)
        start_direction = random.choice(dirs)
        end_direction = random.choice(dirs)
        start = (start_pose[0], start_pose[1], start_direction)
        end = (end_pose[0], end_pose[1], end_direction)
        poses_and_goals.append((f"robot_{n}", start, end))

    return poses_and_goals


def plot_danger_area(map_object: RoadMap, danger_area: Polygon, title: str = ""):
    _, ax = plt.subplots()
    exterior_polygon, interior_polygons = map_object.shapely_to_mplpolygons(
        map_object.map
    )
    ax.add_patch(exterior_polygon)
    for hole_polygon in interior_polygons:
        ax.add_patch(hole_polygon)
    mpl_polygon = MplPolygon(
        list(danger_area.exterior.coords),
        facecolor="red",  # Danger areas are typically red
        alpha=0.3,  # Transparency for visualization
        edgecolor="darkred",
    )
    ax.add_patch(mpl_polygon)
    ax.set_title(f"Danger Area: {title}", fontsize=14)
    min_x, min_y, max_x, max_y = danger_area.bounds
    margin = 1.0  # 1 meter margin
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal", adjustable="box")
    plt.show()


if __name__ == "__main__":
    """
    Test sim_get_poses_and_goals
    """
    # print(sim_get_poses_and_goals(3))
