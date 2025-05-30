"""
Road Network Behaviour Tree
"""

from typing import Optional, List, Dict, Union, Tuple, Callable
import copy

import py_trees
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import substring
import networkx as nx
import logging
from datetime import datetime

from em_vehicle_control.helper_classes.robot_fsm import RobotFSM
from em_vehicle_control.helper_classes.vehicles import *
from em_vehicle_control.helper_classes.map import RoadSegment, RoadMap, RoadTrack
from em_vehicle_control.helper_classes.pathplanners.rrt import RRT_star_Reeds_Shepp

from em_vehicle_control.helper_classes.BThelper import *  # all helper functions

AllVehicles = Union[Edison, EdyMobile]
PosePt2D = Tuple[float, float, float]
PosePathPoint = Tuple[float, float, int]  # x, y, direction

"""
All constants, units attached
"""
TICK_RATE = 0.5  # Hz
DANGER_AREA_BUFFER = 0.01  # m
DANGER_AREA_SWEEP_LOOKAHEAD = 1.1  # times of tick period
"""
Danger area sweep distance is given by 
DANGER_AREA_SWEEP_LOOKAHEAD / TICK_RATE * nominal speed of the robot
"""
NEAR_TO_GOAL_RADIUS = 0.05
NEAR_TO_GOAL_ANGLE = 0.26 # rad or 15 deg
RRT_GOAL_RADIUS = 0.05  # m
NEAR_TO_PATH_RADIUS = 0.08  # m
NEAR_TO_PATH_ANGLE = 0.8  # rad or 45 deg, high just to ensure direction is correct
NEAR_TO_PATH_END_RADIUS = 0.08
NEAR_TO_PATH_END_ANGLE = 0.8  # rad
NEAR_TO_GOAL_USE_SAMPLING = 0.4  # m
REEDS_SHEPP_STEP_LARGE = 0.1  # m
REEDS_SHEPP_STEP_SMALL = 0.01  # m

blackboard = Blackboard()
"""
Blackboard items:
    num_robot: total number of robots on track, moving or otherwise
    robot_names: list of robot names
    robot_FSMs: dict of robot name to robot finite state machine
    robot_types: dict of robot name to vehicle class object
    robot_poses_goals: dict of robot names to: pose, goal
    robot_paths: dict of robot names to: pose_path, node_path, geometry_path
    graph_network(RoadTrack): use full_graph to access nx.graph object
    static_obstacles(List[Polygon]): list of static obstacles
    blocked_nodes(List[int]): list of blocked graph nodes
    danger_areas(Dict[str, Polygon]): danger areas corresponding to each robot
    priority_list(List[str]): List of robot names in order of most priority
    robots_to_move(List[str]): Final list of robots to move
"""


# Configure loggers
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BehaviorTree")

class LoggingBehavior(py_trees.behaviour.Behaviour):
    def update(self):
        status = super().update()
        logger.info(f"Node '{self.name}' Status: {status}")
        if status == py_trees.common.Status.FAILURE:
            self.tree.failure_registry.append({
                "node_name": self.name,
                "timestamp": datetime.now(),
                "parent": self.parent.name if self.parent else None,
            })
        return status


class LoggingDecorator(py_trees.decorators.Decorator):
    def __init__(self, child, name):
        super().__init__(name=name, child=child)
        self.logger = logging.getLogger(name)

    def update(self):
        status = self.decorated.status
        self.logger.info(f"{self.name} status: {status}")
        return status
    

class LoggingLeaf(py_trees.behaviour.Behaviour):
    def update(self):
        status = super().update()
        logging.info(f"Node '{self.name}' status: {status}")
        if status == py_trees.common.Status.FAILURE:
            logging.error(f"Node '{self.name}' failed at {datetime.now()}")
        return status


class LoggingSelector(py_trees.composites.Selector):
    def update(self):
        for child in self.children:
            if child.status == py_trees.common.Status.FAILURE:
                logging.error(f"Child '{child.name}' failed.")
        return super().update()


class CallForHelp(py_trees.behaviour.Behaviour):
    def __init__(self, name="Call for Help, Fix Error"):
        super().__init__(name)

    def update(self):
        print("Action: Calling for help, fixing error...",flush=True)
        return py_trees.common.Status.SUCCESS


class GetAllCurrentPosesAndGoals(py_trees.behaviour.Behaviour):
    def __init__(self, name="Get All Current Poses and Goals"):
        super().__init__(name)

    def at_goal(self, current_pose: PosePt2D, goal: PosePt2D) -> bool:
        """
        Returns:
            -bool: true if near goal
        """
        goal_distance_sq = (current_pose[0] - goal[0]) ** 2 + (current_pose[1] - goal[1]) ** 2
        angle_diff = np.absolute((current_pose[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi)
        if (goal_distance_sq < NEAR_TO_GOAL_RADIUS**2) and (angle_diff < NEAR_TO_GOAL_ANGLE) :
            print("REACHED!", flush=True)
            return True
        return False

    def update_idle_states(self) -> None:
        """
        Update the states of all idle robots with goals to waiting
        """
        robot_poses_goals = blackboard.get("robot_poses_goals")
        robot_fsms = blackboard.get("robot_FSMs")
        for robot_name, data in robot_poses_goals.items():
            goal = data.get("goal")
            if goal != None and robot_fsms[robot_name].state == "idle":
                robot_fsms[robot_name].startup()
            if goal != None and self.at_goal(data.get("pose"), goal):
                robot_fsms[robot_name].to_idle()
        blackboard.set("robot_FSMs", robot_fsms)

    def update(self) -> Status:
        self.update_idle_states()
        return py_trees.common.Status.SUCCESS


def create_fetch_missions_tree():
    root = py_trees.composites.Sequence("Fetch Mission Subtree", memory=False)
    get_all_poses_and_goals = GetAllCurrentPosesAndGoals()
    root.add_child(get_all_poses_and_goals)

    return root


class HandleStaticObstacles(py_trees.behaviour.Behaviour):
    """
    Sets all idle and error robots as static obstacles
    """

    def __init__(self):
        super().__init__("Handle Static Obstacles")
        self.robot_poses_goals = None
        self.robot_types = None
        self.static_obstacles:List[Polygon] = []
        self.blocked_nodes: List[int] = []

    def block_graph_nodes(self, robot_polygon: Polygon) -> None:
        """
        Blocks all nodes, and their corresponding edges,
        on the dynamic_graph as inaccessible.

        Args:
            robot_polygon(Polygon): shape of robot to register
        """
        graph_network:RoadTrack = blackboard.get("graph_network")
        self.blocked_nodes += graph_network.block_nodes_within_obstacle(robot_polygon)

    def update(self) -> Status:
        robot_fsms = blackboard.get("robot_FSMs")
        robot_poses_goals = blackboard.get("robot_poses_goals")
        robot_types = blackboard.get("robot_types")
        for robot_name, fsm in robot_fsms.items():
            if fsm.state == "idle" or fsm.state == "error":
                robot_pose = robot_poses_goals[robot_name]["pose"]
                robot_type = robot_types[robot_name]
                robot_type.construct_vehicle(robot_pose)
                robot_polygon = robot_type.vehicle_model.buffer(DANGER_AREA_BUFFER)
                self.static_obstacles.append(robot_polygon)
                self.block_graph_nodes(robot_polygon)
        blackboard.set("static_obstacles", self.static_obstacles)
        blackboard.set("blocked_nodes", self.blocked_nodes)
        return py_trees.common.Status.SUCCESS


class PlanPathForRobot(py_trees.behaviour.Behaviour):
    """
    Selects planning strategies and sets path for path tracker
    """

    def __init__(self, robot_name: str):
        super().__init__("Plan Path for Robot")
        self.robot_name: str = robot_name
        self.robot_type: AllVehicles = None
        self.robot_fsm: RobotFSM = None
        self.robot_pose: PosePt2D = None
        self.robot_goal: PosePt2D = None
        self.pose_path: List[PosePathPoint] = None
        self.node_path: List[int] = None
        self.geometry_path: List[LineString] = None
        self.graph_network: RoadTrack = None

    @staticmethod
    def make_l2_heuristic(graph: nx.Graph) -> Callable:
        """
        Heuristic for A* algorithm

        Args:
            graph (nx.Graph): The graph on which the heuristic will be applied. Each node
                            in the graph must have a "pos" attribute containing its
                            (x, y) coordinates as a tuple.

        Returns:
            Callable[[int, int], float]: A heuristic function that calculates the
                                        Euclidean distance between two nodes `u` and `v`
                                        based on their "pos" attributes.
        """

        def l2_heuristic(u, v):
            pos_u = graph.nodes[u]["pos"]
            pos_v = graph.nodes[v]["pos"]
            return np.sqrt((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)

        return l2_heuristic

    def plan_Astar(self, source: int, goal: int) -> None:
        """
        Runs Astar algorithm

        Args:
            source(int): Source node
            goal(int): Goal node
        """
        heuristic = self.make_l2_heuristic(self.graph_network.dynamic_graph)
        self.node_path = nx.astar_path(
            self.graph_network.dynamic_graph, source, goal, heuristic, weight="weight"
        )
        self.geometry_path = self.convert_nodes_to_geometry_path(
            self.graph_network, self.node_path
        )
        self.pose_path = self.convert_to_nodes_to_pose_path(self.geometry_path)
        self.set_path()

    def convert_nodes_to_geometry_path(
        self, graph_network: RoadTrack, node_path: List[int]
    ) -> List[LineString]:
        geometry_path = []
        for u, v in zip(node_path, node_path[1:]):
            edge_data = graph_network.dynamic_graph.edges[(u, v)]
            geometry_path.append(edge_data["geometry"])
        return geometry_path

    def convert_to_nodes_to_pose_path(
        self,
        geometry_path: List[LineString],
    ) -> List[PosePathPoint]:
        """
        Convert a sequence of nodes and their associated geometries into a pose path.

        Args:
            geometry_path (List[LineString]): Geometric path (LineStrings between nodes).

        Returns:
            List[Tuple[float, float, int]]: Pose path represented as (x, y, gear).
        """
        pose_path = []
        NUM_SEGMENT_DIRECTION_LOOKAHEAD = 5
        gear = 1
        for seg_idx, segment in enumerate(geometry_path):
            if seg_idx < NUM_SEGMENT_DIRECTION_LOOKAHEAD:
                gear = self.determine_gear_from_geometry_path(segment)
            for x, y in segment.coords:
                pose_path.append((x, y, gear))

        return pose_path

    def determine_gear_from_geometry_path(self, segment) -> int:
        """
        Determine if the robot is moving forward or backward based on the first geometry segment.

        Returns:
            int: 1 if forward gear, -1 if reverse gear.
        """
        start_point = segment.coords[0]
        end_point = segment.coords[-1]
        segment_vector = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])
        segment_vector /= np.linalg.norm(segment_vector)
        robot_direction = np.array([np.cos(self.robot_pose[2]), np.sin(self.robot_pose[2])])
        dot_product = np.dot(segment_vector, robot_direction)
        return 1 if dot_product > 0 else -1

    def plan_rrt_star(self, source: PosePt2D, goal: PosePt2D) -> None:
        """
        Runs RRT* algorithm with Reeds Shepp pathing

        Args:
            source(PosePt2D): current pose of the robot
            goal(PosePt2D): desired pose of the robot
        """
        if not blackboard.exists("static_obstacles"):
            static_obstacles = []
            blackboard.set("static_obstacles", static_obstacles)
        static_obstacles = blackboard.get("static_obstacles")

        local_map = self.graph_network.road_map.get_local_map(
            [Point(source[0], source[1]), Point(goal[0], goal[1])]
        )
        self.pose_path = RRT_star_Reeds_Shepp.create_and_plan(
            source, goal, RRT_GOAL_RADIUS, self.robot_type, local_map, static_obstacles
        )
        self.node_path, self.geometry_path = None, None
        self.set_path()

    def set_path(self) -> None:
        """
        Sets pose, node, and geometry path on the blackboard
        """
        robot_paths = blackboard.get("robot_paths")
        robot_paths[self.robot_name] = {
            "pose_path": self.pose_path,
            "node_path": self.node_path,
            "geometry_path": self.geometry_path,
        }
        blackboard.set("robot_paths", robot_paths)

    def near_to_point(
        self,
        robot_pose: PosePt2D,
        near_to_radius: float,
        near_to_angle: Optional[float] = None,
        point: Optional[PosePt2D] = None,
        point_AB: Optional[Tuple[PosePathPoint, PosePathPoint]] = None,
    ) -> bool:
        """
        Checks if robot_pose is near to a point, or a point_A and uses point_B to get an angle at point_A.
        Point_B points to point_A, or the path is [..., point_B, point_A]
        Either point or point_AB must be filled.

        Args:
            robot_pose (PosePt2D): robot pose
            near_to_radius (float): acceptable radius for closeness
            near_to_angle(Optional[float]): acceptable angle for closeness
            point (Optional[PosePt2D]): point on path to determine closeness
            point_AB (Optional[Tuple[PosePathPoint, PosePathPoint]]): 2 points to determine closeness at point A and angle with point B
        """
        if (point is None) == (point_AB is None):
            raise ValueError(
                "Provide only one of 'point' or 'point_AB', not both or none."
            )

        if near_to_angle is not None:
            if point is not None:
                target_angle = point[2]
            else:
                target_angle = np.arctan2(
                    point_AB[0][1] - point_AB[1][1], point_AB[0][0] - point_AB[1][0]
                )
                if point_AB[0][2] == -1:  # reverse gear needed
                    target_angle = (target_angle + 2 * np.pi) % 2 * np.pi - np.pi
            if (
                (target_angle - robot_pose[2] + np.pi) % 2 * np.pi - np.pi
            ) > near_to_angle:
                return False

        if point is not None:
            target_x, target_y = point[0], point[1]
        else:
            target_x, target_y = point_AB[0][0], point_AB[0][1]
        dist = np.sqrt(
            (target_x - self.robot_pose[0]) ** 2 + (target_y - self.robot_pose[1]) ** 2
        )
        if dist > near_to_radius:
            return False
        return True

    def path_nearly_finished(self) -> bool:
        """
        Checks if path is nearly finished.
        Uses pose_path, NEAR_TO_PATH_END_RADIUS, NEAR_TO_PATH_END_ANGLE

        Returns:
            bool: True if path is nearly finished
        """
        last_pose = self.pose_path[-1]
        second_last_pose = self.pose_path[-2]
        return self.near_to_point(
            self.robot_pose,
            NEAR_TO_PATH_END_RADIUS,
            NEAR_TO_PATH_END_ANGLE,
            point_AB=(last_pose, second_last_pose),
        )

    def near_to_goal(self) -> bool:
        """
        Checks if robot is near to its goal, to use sampling algorithm
        Uses NEAR_TO_GOAL_USE_SAMPLING

        Returns:
            bool: True if near to goal
        """
        return self.near_to_point(
            self.robot_pose,
            NEAR_TO_GOAL_USE_SAMPLING,
            point=self.robot_goal,
        )

    def near_to_graph(self) -> Tuple[bool, int, PosePt2D]:
        """
        Checks if robot is near to a graph node
        Uses NEAR_TO_PATH_RADIUS, NEAR_TO_PATH_ANGLE

        Note: The logic here is not intuitive. It will always return the nearest unobstructed graph node.
        Then the bool will return true only if both distance and angle are in acceptable limits.

        Returns:
            Tuple:
                - bool: True if near to a free node
                - int: nearest free node
                - PosePt2D: graph node pose with angle of the path beginning
        """
        if not blackboard.exists("blocked_nodes"):
            blocked_nodes = []
            blackboard.set("blocked_nodes", blocked_nodes)
        blocked_nodes = blackboard.get("blocked_nodes")
        robot_current_point = (self.robot_pose[0], self.robot_pose[1])
        nearby_nodes_idxs = self.graph_network.get_N_nearest_vertices(
            robot_current_point, len(blocked_nodes) + 1
        )  # returns at least 1 more than blocked nodes
        goal_node_idx = self.graph_network.get_N_nearest_vertices(
            (self.robot_goal[0], self.robot_goal[1])
        )[0]
        for nearby_node_idx in nearby_nodes_idxs:
            if nearby_node_idx in blocked_nodes:
                continue
            node_point = self.graph_network.dynamic_graph.nodes[nearby_node_idx]["pos"]
            path = nx.astar_path(
                self.graph_network.dynamic_graph,
                nearby_node_idx,
                goal_node_idx,
                self.make_l2_heuristic(self.graph_network.dynamic_graph),
            )
            if nearby_node_idx == goal_node_idx:
                # edge case, do RRT to goal
                return False, None, (*node_point, self.robot_goal[2])
            second_node_point = self.graph_network.dynamic_graph.nodes[path[1]]["pos"]
            target_angle = np.arctan2(
                second_node_point[1] - node_point[1],
                second_node_point[0] - node_point[0],
            )  # node_point points to second_node_point
            dist = np.sqrt(
                (node_point[0] - self.robot_pose[0]) ** 2
                + (node_point[1] - self.robot_pose[1]) ** 2
            )
            is_near = (
                dist < NEAR_TO_PATH_RADIUS
                and (target_angle - self.robot_pose[2] + np.pi) % 2 * np.pi - np.pi
                < NEAR_TO_PATH_ANGLE
            )
            return is_near, nearby_node_idx, (*node_point, target_angle)

    def get_new_path(self) -> None:
        """
        Gets new path by checking if near to goal or graph
        """
        if self.near_to_goal():
            self.plan_rrt_star(self.robot_pose, self.robot_goal)
            self.robot_fsm.resume_sampling()
            return
        is_near_to_graph, nearest_node_idx, nearest_node_pose = self.near_to_graph()
        if is_near_to_graph:
            goal_node_idx = self.graph_network.get_N_nearest_vertices(
                (self.robot_goal[0], self.robot_goal[1])
            )[0]
            self.plan_Astar(nearest_node_idx, goal_node_idx)
            self.robot_fsm.resume_graph()
            return
        else:
            self.plan_rrt_star(self.robot_pose, nearest_node_pose)
            self.robot_fsm.resume_sampling()

    def near_to_previous_pose_path(self) -> bool:
        """
        Checks if near to previous pose path.
        Note: Can only be used on Reeds Shepp paths. RoadTrack poses are too sparse.

        Returns:
            bool: true if near to previous path
        """
        for idx, pose_point in enumerate(self.pose_path):
            dist_squared = (self.robot_pose[0] - pose_point[0]) ** 2 + (
                self.robot_pose[1] - pose_point[1]
            ) ** 2
            if dist_squared < NEAR_TO_PATH_RADIUS ** 2:
                return True
        return False

    def trim_path(self, nearest_seg_idx: int) -> None:
        """
        Trims node, geometry and pose paths so that the robot is on the first segment

        Args:
            nearest_seg_idx: index with the nearest segment
        """
        self.geometry_path = self.geometry_path[nearest_seg_idx:]
        self.node_path = self.node_path[nearest_seg_idx:]
        self.pose_path = self.convert_to_nodes_to_pose_path(self.geometry_path)
        self.set_path()
    
    def near_to_previous_geometry_path(self) -> bool:
        """
        Checks if near to previous geometry path

        Returns:
            bool: true if near to previous path
        """
        point = Point(self.robot_pose[0], self.robot_pose[1])
        for index, segment in enumerate(self.geometry_path):
            projected_distance = segment.project(point)
            candidate_point = segment.interpolate(projected_distance)
            distance = point.distance(candidate_point)
            if distance < NEAR_TO_PATH_RADIUS:
                self.trim_path(index)
                return True
        return False

    def near_to_previous_path(self) -> bool:
        """
        Checks if robot is near to previous path
        
        Returns:
            bool: true if near to previous path
        """
        if self.geometry_path is not None:
            return self.near_to_previous_geometry_path()
        elif self.pose_path is not None:
            return self.near_to_previous_pose_path()
        else:
            return False
        # if self.robot_state == "move_by_sampling":
        #     return self.near_to_previous_pose_path()
        # else:
        #     return self.near_to_previous_geometry_path()

    def resume_motion(self) -> None:
        """
        Resumes motion if robot was waiting and has a geometry or pose graph
        Note: Does not work because dynamic graph has changed
        """
        if self.geometry_path is not None:
            if self.near_to_previous_geometry_path():
                self.robot_fsm.resume_graph()
                return
            else:
                self.get_new_path()
                return
        elif self.pose_path is not None:
            if self.near_to_previous_pose_path():
                self.robot_fsm.resume_sampling()
                return
            else:
                self.get_new_path()
                return
        self.get_new_path()
        return
    
    def determine_path(self) -> bool:
        if (
            self.robot_fsm.state == "move_by_sampling"
            or self.robot_fsm.state == "move_by_graph"
        ):
            if self.pose_path is None or self.path_nearly_finished():
                # print("Determine path", self.robot_name, "path nearly finished, get new path", flush=True)
                self.get_new_path()
                return True
            elif self.near_to_previous_path():
                # continue on previous path
                # print("Determine path", self.robot_name, "continuing on previous path", flush=True)
                return True
            else:
                # print("Determine path", self.robot_name, "bad previous path, getting new path", flush=True)
                self.get_new_path()
                return True
        elif self.robot_fsm.state == "waiting":
            # print("Determine path", self.robot_name, "back from waiting, get new path", flush=True)
            self.get_new_path()
            return True
        # unknown state of system, returns failure
        return False

    def update(self) -> Status:
        robot_poses_goals = blackboard.get("robot_poses_goals")
        robot_pose_goal = robot_poses_goals[self.robot_name]
        self.robot_pose = robot_pose_goal["pose"]
        self.robot_type = blackboard.get("robot_types")[self.robot_name]
        self.robot_goal = robot_pose_goal["goal"]
        robot_fsms = blackboard.get("robot_FSMs")
        self.robot_fsm = robot_fsms[self.robot_name]
        if self.robot_fsm.state == "idle":
            return py_trees.common.Status.SUCCESS
        if not blackboard.exists("robot_paths"):
            robot_paths = {}
            blackboard.set("robot_paths", robot_paths)
        robot_paths = blackboard.get("robot_paths")
        robot_paths = robot_paths.get(self.robot_name)
        if robot_paths is None:
            robot_paths = {}
        self.pose_path = robot_paths.get("pose_path")
        self.geometry_path = robot_paths.get("geometry_path")
        self.node_path = robot_paths.get("node_path")
        self.graph_network = blackboard.get("graph_network")

        if self.determine_path():
            return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.FAILURE


def create_plan_paths_tree(robot_names: List[str]) -> py_trees.composites.Sequence:
    """
    Creates the Plan Paths tree for the fleet.

    Args:
        robot_names (List[str]): List of robot names.

    Returns:
        py_trees.composites.Sequence: The full Plan Paths tree.
    """
    root = py_trees.composites.Sequence("Plan Paths Subtree", memory=False)
    handle_static_obstacles = HandleStaticObstacles()
    root.add_child(handle_static_obstacles)

    # Add behaviors for planning paths for individual robots
    plan_path_nodes = []
    for robot_name in robot_names:
        plan_path_nodes.append(PlanPathForRobot(robot_name))

    # Add all plan path behaviors as children
    root.add_children(plan_path_nodes)
    return root


class ComputeDangerAreaForRobot(py_trees.behaviour.Behaviour):
    def __init__(self, robot_name: str):
        super().__init__(f"Compute Danger Area for {robot_name}")
        self.robot_name = robot_name
        self.robot_type = None

    def setup(self) -> None:
        self.robot_type: AllVehicles = blackboard.get("robot_types")[self.robot_name]

    def get_upcoming_path(
        self,
        current_pose: Point,
        closest_point: Point,
        closest_seg_index: int,
        path: List[LineString],
        distance: float,
    ) -> LineString:
        """
        Calculates and returns the upcoming path from the current pose of specified distance
        Args:
            current_pose(Point): current pose of robot
            closest_point(Point): closest point on path to robot
            closest_seg_index(int): index of the segment which the closest point falls on
            path(List[Linestring]): full path of robot
            distance(float): in metres, the distance along path, starting from closest_point
        Returns:
            (LineString): The upcoming linestring path
        """
        remaining_distance = distance
        upcoming_coords = []

        # Add the segment from current_pose to closest_point
        segment_to_closest = LineString([current_pose, closest_point])
        segment_length = segment_to_closest.length
        if segment_length >= remaining_distance:
            # If this segment alone exceeds the remaining distance, clip it and return
            result = substring(segment_to_closest, 0, remaining_distance)
            return result

        upcoming_coords.extend(segment_to_closest.coords)
        remaining_distance -= segment_length

        # Traverse the path starting from closest_seg_index
        for index in range(closest_seg_index, len(path)):
            segment = path[index]
            if index == closest_seg_index:
                # For the first segment, start from the closest point
                start_dist = segment.project(closest_point)
                end_dist = min(start_dist + remaining_distance, segment.length)
                segment_part = substring(segment, start_dist, end_dist)
            else:
                # For subsequent segments, start from the beginning
                end_dist = min(remaining_distance, segment.length)
                segment_part = substring(segment, 0, end_dist)

            upcoming_coords.extend(segment_part.coords)
            remaining_distance -= segment_part.length

            if remaining_distance <= 10e-5:
                # Clip the final segment to fit the exact remaining distance
                excess_length = abs(remaining_distance)
                final_segment = substring(
                    segment_part, 0, segment_part.length - excess_length
                )
                upcoming_coords = upcoming_coords[: -(len(segment_part.coords))]
                upcoming_coords.extend(final_segment.coords)
                break
        
        return LineString(upcoming_coords)
    
    def find_nearest_point_on_geom_path(
        self, point: Point, path: List[LineString]
    ) -> Tuple[Point, int]:
        """
        Find the nearest point on a geometric path (list of linestrings), to a given point

        Args:
            point(Point): Given point
            path(List[LineString]): Geometric path
        Returns:
            Tuple[Point, int]:
                - nearest_point(Point): Coordinates of the closest point lying on the path
                - index(int): index of LineString segment the nearest_point lies on
        """
        closest_point = None
        closest_segment_index = -1
        min_distance = float("inf")

        for index, segment in enumerate(path):
            projected_distance = segment.project(point)
            candidate_point = segment.interpolate(projected_distance)
            distance = point.distance(candidate_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = candidate_point
                closest_segment_index = index

        return closest_point, closest_segment_index

    def update(self) -> py_trees.common.Status:
        robot_pose = blackboard.get("robot_poses_goals")[self.robot_name]["pose"]
        self.robot_type.construct_vehicle(robot_pose)
        robot_footprint = self.robot_type.vehicle_model
        robot_fsm = blackboard.get("robot_FSMs")[self.robot_name]

        danger_area = None
        if robot_fsm.state in ("idle", "waiting", "error"):
            danger_area = robot_footprint.buffer(DANGER_AREA_BUFFER)
        elif robot_fsm.state == "move_by_sampling":
            path = copy.deepcopy(
                blackboard.get("robot_paths")[self.robot_name]["pose_path"]
            )
            if not path:
                print("Compute danger area error, no path planned while in state move_sampling", flush=True)
                return py_trees.common.Status.FAILURE
            path = [(x, y) for x, y, _ in path]
            bounds = robot_footprint.bounds
            width = bounds[2] - bounds[0]
            buffer_distance = width / 2
            line = LineString(path)
            danger_area = line.buffer(buffer_distance + DANGER_AREA_BUFFER)
        elif robot_fsm.state == "move_by_graph":
            path = copy.deepcopy(
                blackboard.get("robot_paths")[self.robot_name]["geometry_path"]
            )
            if not path:
                print("Compute danger area error, no path planned while in state move_graph", flush=True)
                return py_trees.common.Status.FAILURE
            current_pose = Point(robot_pose[0], robot_pose[1])
            nearest_point, closest_seg_idx = self.find_nearest_point_on_geom_path(
                current_pose, path
            )
            lookahead_dist = (
                DANGER_AREA_SWEEP_LOOKAHEAD / TICK_RATE * self.robot_type.nominal_speed
            )
            line = self.get_upcoming_path(
                current_pose, nearest_point, closest_seg_idx, path, lookahead_dist
            )
            bounds = robot_footprint.bounds
            width = bounds[2] - bounds[0]
            danger_area = line.buffer(width / 2 + DANGER_AREA_BUFFER)
        if blackboard.exists("danger_areas"):
            danger_areas = blackboard.get("danger_areas")
        else:
            danger_areas = {}
        danger_areas[self.robot_name] = danger_area
        blackboard.set("danger_areas", danger_areas)
        return py_trees.common.Status.SUCCESS


def create_compute_danger_areas_tree(robot_names: List[str]):
    root = py_trees.composites.Sequence("Compute Danger Areas Subtree", memory=False)

    compute_danger_area_subtree_nodes = []
    for robot_name in robot_names:
        compute_danger_area_subtree_nodes.append(ComputeDangerAreaForRobot(robot_name))
    root.add_children(compute_danger_area_subtree_nodes)

    return root


class DetermineRobotsPriority(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Determine Robots Priority")
        self.robot_names: List[str] = None
        self.robot_fsms: Dict[str, RobotFSM] = None
        self.danger_areas: Dict[str, Polygon] = None
        self.robot_types: Dict[str, AllVehicles] = None
        self.robot_poses_goals: Dict[str, Dict] = None
        self.static_obstacles: List[Polygon] = None
        self.robot_paths: Dict[str, Dict] = None
        self.graph_network: RoadTrack = None

        self.intersections_reserved: Dict[Polygon:str] = None

    def goal_not_blocked(self, robot_A: str, robot_B: str) -> bool:
        """
        Checks if robot_B's goal is blocked by robot_A's footprint

        Args:
            robot_A (str): name of robot_A
            robot_B (str): name of robot_B
        Returns:
            bool: True if robot_A does not block robot_B's goal
        """
        robot_B_old_pose_path = self.robot_paths[robot_B]["pose_path"]
        robot_B_last_pose = robot_B_old_pose_path[-1]
        goal = Point(robot_B_last_pose[0], robot_B_last_pose[1])
        robot_A_pose = self.robot_poses_goals[robot_A]["pose"]
        self.robot_types[robot_A].construct_vehicle(robot_A_pose)
        robot_A_footprint = self.robot_types[robot_A].vehicle_model
        robot_A_obstacle = robot_A_footprint.buffer(DANGER_AREA_BUFFER)

        return not robot_A_obstacle.contains(goal)
    
    def get_final_pose(self, last_pose_pt:PosePathPoint, second_last_pose_pt: PosePathPoint)->PosePt2D:
        """
        Returns the final pose with angle, robot is moving from second last to last point.
        """
        angle = np.arctan2(last_pose_pt[1]-second_last_pose_pt[1], last_pose_pt[0]-second_last_pose_pt[0])
        if last_pose_pt[2] == -1:
            angle += np.pi
            angle = (angle + np.pi) % 2*np.pi - np.pi
        return (last_pose_pt[0], last_pose_pt[1], angle)

    def replan_sampling(self, robot_A: str, robot_B: str) -> None:
        """
        Replans robot_B such that it does not collide with robot_A.
        Sets robot_A to waiting, but at lower priority than robot_B

        Args:
            robot_A (str): name of robot_A
            robot_B (str): name of robot_B
        """
        self.robot_fsms[robot_A].wait()
        robot_A_pose = self.robot_poses_goals[robot_A]["pose"]
        robot_B_pose = self.robot_poses_goals[robot_B]["pose"]
        self.robot_types[robot_A].construct_vehicle(robot_A_pose)
        robot_A_footprint = self.robot_types[robot_A].vehicle_model
        robot_A_obstacle = robot_A_footprint.buffer(DANGER_AREA_BUFFER)
        robot_B_old_pose_path = self.robot_paths[robot_B]["pose_path"]
        goal = self.get_final_pose(robot_B_old_pose_path[-1], robot_B_old_pose_path[-2])
        local_map = self.graph_network.road_map.get_local_map([(robot_B_pose[0],robot_B_pose[1]), (goal[0], goal[1])])
        pose_path = RRT_star_Reeds_Shepp.create_and_plan(
            robot_B_pose,
            goal,
            RRT_GOAL_RADIUS,
            self.robot_types[robot_B],
            local_map,
            self.static_obstacles + [robot_A_obstacle],
        )
        robot_B_paths = {
            "pose_path": pose_path,
            "node_path": None,
            "geometry_path": None
        }
        self.robot_paths[robot_B] = robot_B_paths
        blackboard.set("robot_paths", self.robot_paths)

    def prioritise_using_lane_rules(self, robot_A: str, robot_B: str) -> bool:
        """
        Checks the priority between robot_A and robot_B using lanes

        Args:
            robot_A (str): name of robot_A
            robot_B (str): name of robot_B
        Returns:
            bool: True if robot_A has higher priority than robot_B
        """
        robot_A_node_path = self.robot_paths[robot_A]["node_path"]
        robot_B_node_path = self.robot_paths[robot_B]["node_path"]
        nodes = self.graph_network.dynamic_graph.nodes
        print("using lane rules", flush=True)
        # robot_A_stays_on_same_lane = (nodes[robot_A_current_node]["lane"] == nodes[robot_A_next_node]["lane"])
        # robot_B_stays_on_same_lane = (nodes[robot_B_current_node]["lane"] == nodes[robot_B_next_node]["lane"])
        # if robot_A_stays_on_same_lane != robot_B_stays_on_same_lane:
        #     return robot_A_stays_on_same_lane # the robot that stays on the same lane has priority
        if nodes[robot_A_node_path[0]]["lane"] == nodes[robot_B_node_path[0]]["lane"]:
            print("yes same lane", flush=True)
            return self.prioritise_robots_on_same_lane(robot_A, robot_B)
        else:
            print("no same lane", flush=True)
            return self.prioritise_robots_on_different_lanes(robot_A, robot_B)
        
    def prioritise_robots_on_same_lane(self, robot_A: str, robot_B: str) -> bool:
        """
        Checks the priority between robot_A and robot_B on the same lane

        Args:
            robot_A (str): name of robot_A
            robot_B (str): name of robot_B
        Returns:
            bool: True if robot_A has higher priority than robot_B
        """
        print("Prioritising on same lane", flush=True)
        robot_A_node_path = self.robot_paths[robot_A]["node_path"]
        robot_A_pose_path = self.robot_paths[robot_A]["pose_path"]
        robot_B_pose_path = self.robot_paths[robot_B]["pose_path"]
        if robot_A_pose_path[0][2] != robot_B_pose_path[0][2]:
            # for their DA to overlap AND
            # both robots to be in the same lane AND
            # their direction to be opposite
            # collision is imminent
            self.robot_fsms[robot_A].lose_control()
            self.robot_fsms[robot_B].lose_control()
            blackboard.set("robot_FSMs", self.robot_fsms)
            return True
        nodes = self.graph_network.dynamic_graph.nodes
        robot_A_pose = self.robot_poses_goals[robot_A]["pose"]
        robot_B_pose = self.robot_poses_goals[robot_B]["pose"]
        lane_direction = nodes[robot_A_node_path[0]]["lane"]
        if lane_direction == "north":
            # larger y has priority
            return robot_A_pose[1] > robot_B_pose[1]
        elif lane_direction == "south":
            # smaller y has priority
            return robot_A_pose[1] < robot_B_pose[1] 
        elif lane_direction == "east":
            # larger x has priority
            return robot_A_pose[0] > robot_B_pose[0] 
        else: # west
            # smaller x has priority
            return robot_A_pose[0] < robot_B_pose[0]  

    def prioritise_robots_on_different_lanes(self, robot_A: str, robot_B: str) -> bool:
        """
        Checks the priority between robot_A and robot_B on different lanes

        Args:
            robot_A (str): name of robot_A
            robot_B (str): name of robot_B
        Returns:
            bool: True if robot_A has higher priority than robot_B
        """
        print("Prioritising on different lane", flush=True)
        danger_area_A = self.danger_areas[robot_A]
        danger_area_B = self.danger_areas[robot_B]
        danger_area_overlap = danger_area_A.intersection(danger_area_B)
        intersections = self.graph_network.road_intersections
        for intersection in intersections:
            if danger_area_overlap.overlaps(intersection):
                return False # maintain old priority
        else:

            # print("Lane collision: danger area outside of intersection. Unknown situation.", flush=True)
            # self.robot_fsms[robot_A].lose_control()
            # self.robot_fsms[robot_B].lose_control()
            # blackboard.set("robot_FSMs", self.robot_fsms)
            return True
        
    def intersection_priority(self, robot_A:str, robot_B:str) -> Optional[bool]:
        """
        TODO: should check if danger area overlap is in the intersection too, if it is not, then it should not follow intersection rules
        Returns:
            bool: True if robot_A has higher priority, False if robot_B has higher priority,
                None if priority is undecided
        """
        for value in self.intersections_reserved.values():
            if value == "free_intersection":
                continue
            else:
                if value["reserved_for"] == robot_A and robot_B in value["waiting"]:
                    self.robot_fsms[robot_B].wait()
                    return True
                elif value["reserved_for"] == robot_B and robot_A in value["waiting"]:
                    self.robot_fsms[robot_A].wait()
                    return False
        else:
            return None

    def reserve_intersection(self, robot:str) -> None:
        """
        Reserve free intersection if overlaps with danger area
        """
        for intersection, value in self.intersections_reserved.items():
            if self.danger_areas[robot].overlaps(intersection):
                if value != "free_intersection":
                    if value["reserved_for"] == robot or robot in value["waiting"]:
                        return
                    value["waiting"].append(robot)
                    return
                else:
                    # Reserve the intersection for the robot.
                    self.intersections_reserved[intersection] = {"reserved_for": robot, "waiting": []}
                return
        return

    def manage_intersections(self) -> None:
        """
        Checks all intersections,
        if the robot's danger area intersects an intersection,
        reserve it if it is free.
        """
        # first, free all cleared intersection
        for crossing, value in self.intersections_reserved.items():
            if value != "free_intersection":
                # print("is stll in intersection?", self.danger_areas[value["reserved_for"]].overlaps(crossing), flush=True)
                def plot_polygon(ax, polygon, color, label):
                    if not polygon.is_empty:
                        x, y = polygon.exterior.xy
                        ax.fill(x, y, alpha=0.5, fc=color, label=label)
                fig, ax = plt.subplots()
                plot_polygon(ax, self.danger_areas[value["reserved_for"]], 'blue', 'Polygon 1')
                plot_polygon(ax, crossing, 'green', 'Polygon 2')
                ax.legend()
                ax.set_title("Shapely Polygons Visualization")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.grid(True)
                # plt.show()
            if (
                value != "free_intersection" and
                not self.danger_areas[value["reserved_for"]].overlaps(crossing)
            ):
                # robot no longer overlaps intersection
                self.intersections_reserved[crossing] = "free_intersection"
        for robot in self.robot_names:
            self.reserve_intersection(robot)

    def has_higher_priority(self, robot_A: str, robot_B: str) -> bool:
        """
        Checks the priority between robot_A and robot_B

        Args:
            robot_A (str): name of robot_A
            robot_B (str): name of robot_B
        Returns:
            bool: True if robot_A has higher priority than robot_B
        """
        state_A = self.robot_fsms[robot_A].state
        state_B = self.robot_fsms[robot_B].state
        
        danger_area_A = self.danger_areas[robot_A]
        danger_area_B = self.danger_areas[robot_B]
        intersection_result = self.intersection_priority(robot_A, robot_B)
        if intersection_result is not None:
            return intersection_result
        if not danger_area_A.intersects(danger_area_B):
            # no intersection, no priority change
            return False
        
        if state_B in ("idle", "waiting", "error"):
            # set A to wait, decide on movement next tick
            self.robot_fsms[robot_A].wait()
            blackboard.set("robot_FSMs", self.robot_fsms)
            return False
        if state_A in ("idle", "waiting", "error"):
            # A is waiting, it has priority
            self.robot_fsms[robot_B].wait()
            blackboard.set("robot_FSMs", self.robot_fsms)
            return True
        if state_A == state_B == "move_by_sampling":
            if self.goal_not_blocked(robot_A, robot_B):
                self.replan_sampling(robot_A, robot_B)  # TODO check if covers goal!
                return False
            elif self.goal_not_blocked(robot_B, robot_A):
                self.replan_sampling(robot_B, robot_A)
                return True
            else:
                self.robot_fsms[robot_A].lose_control()
                self.robot_fsms[robot_B].lose_control()
                blackboard.set("robot_FSMs", self.robot_fsms)
                return True # both robots in error state
        if state_A == "move_by_sampling" and state_B == "move_by_graph":
            # sampling method has priority
            return True
        if state_B == "move_by_sampling" and state_A == "move_by_graph":
            return False
        if state_A == state_B == "move_by_graph":
            return self.prioritise_using_lane_rules(robot_A, robot_B)

    def determine_priority(self, robot_name: str) -> None:
        """robot_fsms
        Slots robot into its correct priority
        As long as robot has higher priority, it would be slot into the list.
        Else, it is appended at the end.
        """
        if self.priority_list == []:
            self.priority_list.append(robot_name)
            return
        for index, prioritised_robot in enumerate(self.priority_list):
            if self.has_higher_priority(robot_name, prioritised_robot):
                self.priority_list.insert(index, robot_name)
                return
        else:
            self.priority_list.append(robot_name)

    def initialise(self):
        self.priority_list:list[str] = []
        return super().initialise()

    def update(self) -> Status:
        self.robot_names = blackboard.get("robot_names")
        self.robot_fsms = blackboard.get("robot_FSMs")
        self.danger_areas = blackboard.get("danger_areas")
        self.robot_types = blackboard.get("robot_types")
        self.robot_poses_goals = blackboard.get("robot_poses_goals")
        self.static_obstacles = blackboard.get("static_obstacles")
        self.robot_paths = blackboard.get("robot_paths")
        self.graph_network = blackboard.get("graph_network")

        if self.intersections_reserved is None:
            self.intersections_reserved = {item: "free_intersection" for item in self.graph_network.road_intersections}

        self.manage_intersections()
        for robot_name in self.robot_names:
            self.determine_priority(robot_name)

        blackboard.set("priority_list", self.priority_list)

        return py_trees.common.Status.SUCCESS


class MoveRobots(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Move Robots")

        self.robot_fsms: dict[str, RobotFSM] = None
        self.danger_areas: dict[str, Polygon] = None
        self.robots_to_move: List[str] = None
        self.graph_network: RoadTrack = None

        # self.intersections_reserved = None

    # def reserve_intersection(self, robot:str) -> None:
    #     """
    #     Reserve intersection if robot in intersection
    #     """
    #     if self.robot_fsms[robot].state in ("idle", "waiting", "error"):
    #         return None
    #     for intersection, is_reserved in self.intersections_reserved.items():
    #         if self.danger_areas[robot].overlaps(intersection):
    #             if is_reserved:
    #                 # If the intersection is already reserved, set the robot to wait.
    #                 self.robot_fsms[robot].wait()
    #             else:
    #                 # Reserve the intersection for the robot.
    #                 self.intersections_reserved[intersection] = True
    #             return
    #     return None
    
    def final_move_check(self, priority_list: List[str]):
        """
        Move robot if it does not collide with anything with higher priority
        """
        robots_to_move = []
        for i, robot in enumerate(priority_list):
            robot_DA = self.danger_areas[robot]
            if self.robot_fsms[robot].state in ("idle", "waiting", "error"):
                continue
            if any(robot_DA.overlaps(self.danger_areas[prioritised_robot])
                for prioritised_robot in priority_list[:i]):
                self.robot_fsms[robot].wait()
            else:
                robots_to_move.append(robot)
        blackboard.set("robots_to_move", robots_to_move)

    def update(self) -> Status:
        self.robot_fsms = blackboard.get("robot_FSMs")
        self.danger_areas = blackboard.get("danger_areas")
        self.graph_network = blackboard.get("graph_network")
        # self.intersections_reserved = {item: False for item in self.graph_network.road_intersections}
        priority_list = blackboard.get("priority_list")
        # for robot in priority_list:
        #     self.reserve_intersection(robot)
        self.final_move_check(priority_list)
        return py_trees.common.Status.SUCCESS


def create_prioritise_and_move_robots_tree():
    root = py_trees.composites.Sequence(
        "Prioritise and Move Robots Subtree", memory=False
    )
    root.add_child(DetermineRobotsPriority())
    root.add_child(MoveRobots())
    return root


class FleetManagerTree:
    def __init__(
        self,
        num_robots: int,
        graph_network: RoadTrack,
        vehicle_types: List[AllVehicles],
        robot_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            num_robots(int): Number of robots
            graph_network(RoadTrack): Fully initialised graph of the road network
            vehicle_types(List[AllVehicles]): Type of vehicle, eg. Edison(), use the uninitialised class instance
            robot_names(Optional[List[str]]): Names of robots. If left blank, robots are named robot_0, robot_1...
        """
        self.num_robots = num_robots
        self.graph_network = graph_network
        self.robot_names = robot_names
        if self.robot_names is not None:
            assert self.num_robots == len(self.robot_names)
        else:
            self.robot_names = []
            for i in range(self.num_robots):
                self.robot_names += [f"robot_{i}"]
        self.vehicle_types = vehicle_types

        # self.logger = logging.getLogger("FleetManagerTree")
        self.root = self.create_fleet_manager_tree()
        self.tree = py_trees.trees.BehaviourTree(self.root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG 

    def create_fleet_manager_tree(self):
        root = py_trees.composites.Selector("Fleet Manager", memory=False)
        call_for_help = CallForHelp()
        fetch_missions_tree = create_fetch_missions_tree()
        plan_paths_tree = create_plan_paths_tree(self.robot_names)
        compute_danger_areas_tree = create_compute_danger_areas_tree(self.robot_names)
        prioritise_and_move_robots_tree = create_prioritise_and_move_robots_tree()
        # root = LoggingSelector("Fleet Manager", memory=False)
        # call_for_help = LoggingLeaf(name="Call For Help")
        # fetch_missions_tree = LoggingDecorator(create_fetch_missions_tree(), "Fetch Missions Tree")
        # plan_paths_tree = LoggingDecorator(create_plan_paths_tree(self.robot_names), "Plan Paths Tree")
        # compute_danger_areas_tree = LoggingDecorator(create_compute_danger_areas_tree(self.robot_names), "Compute Danger Areas Tree")
        # prioritise_and_move_robots_tree = LoggingDecorator(create_prioritise_and_move_robots_tree(), "Prioritise and Move Robots Tree")

        main_tasks_sequence = py_trees.composites.Sequence(
            "Main Tasks Sequence", memory=False
        )
        main_tasks_sequence.add_children(
            [
                fetch_missions_tree,
                plan_paths_tree,
                compute_danger_areas_tree,
                prioritise_and_move_robots_tree,
            ]
        )

        root.add_children([main_tasks_sequence, call_for_help])
        return root

    def tick(self):
        self.tree.tick()
        fsms = blackboard.get("robot_FSMs")
        print("State", [x.state for x in fsms.values()], flush=True)

    def get_robots_to_move(self) -> List[str]:
        return blackboard.get("robots_to_move") if blackboard.exists("robots_to_move") else []
    
    def get_pose_path(self, robot:str) -> Optional[List[PosePathPoint]]:
        paths = blackboard.get("robot_paths") if blackboard.exists("robot_paths") else []
        if paths == []:
            return None
        return paths[robot]["pose_path"]
    
    def get_danger_areas(self) -> List[Polygon]:
        danger_areas_dict = blackboard.get("danger_areas")
        return [x for x in danger_areas_dict.values()]

    def set_robot_poses_goals(self, poses_and_goals: Dict) -> None:
        """
        Sets poses and goals
        """
        blackboard.set("robot_poses_goals", poses_and_goals)

    def setup(self):
        """
        Sets up all main functions of the road network behaviour tree including:
            - Robot names
            - Graph network
            - Robot shape (vehicle type)
            - Robot finite state machines
        """
        blackboard.set("num_robots", self.num_robots)
        blackboard.set("robot_names", self.robot_names)

        blackboard.set("graph_network", self.graph_network)

        assert self.num_robots == len(self.vehicle_types)
        vehicles = {}
        for name, vehicle_type in zip(self.robot_names, self.vehicle_types):
            assert not isinstance(  # TODO: just change it to the class instance
                vehicle_type, type
            ), f"{vehicle_type} is a class, not an instance. Did you forget the ()?"
            vehicles[name] = vehicle_type
        blackboard.set("robot_types", vehicles)

        fleet_machines: Dict[str:RobotFSM] = {}
        for name in self.robot_names:
            fsm = RobotFSM(name)
            fleet_machines[name] = fsm
        blackboard.set("robot_FSMs", fleet_machines)

        self.tree.setup(timeout=15)


if __name__ == "__main__":
    test_main_fleet_manager_setup = False
    test_fetch_mission_tree = False
    test_compute_danger_area, visualise_compute_danger_area = False, False
    test_block_static_obstacles = False
    test_path_planning, visualise_path_planning = False, False
    test_prioritise_robots = False
    """
    Set up road network
    """
    test_roads = [
        RoadSegment((2.87, 1.67), (3.52, 13.67)),
        RoadSegment((6.9, 0.3), (7.55, 15.67)),
        RoadSegment((4.72, 1.67), (5.7, 8.64)),
        RoadSegment((2.87, 8), (7.55, 8.64)),
        RoadSegment((2.87, 1.67), (7.55, 2.32)),
    ]
    test_map = RoadMap(test_roads)
    test_graph = RoadTrack(test_map)

    """
    Test main fleet manager tree setup
    """
    if test_main_fleet_manager_setup:
        print("TEST main fleet manager setup")
        FMTree = FleetManagerTree(3, test_graph, [Edison(), Edison(), EdyMobile()])
        FMTree.setup()
        print("  Total number of robots:", blackboard.get("num_robots"))
        print(
            "  Robot names, type and states: ",
            [
                (x[0], y.__class__.__name__, x[1].state)
                for (x, y) in zip(
                    blackboard.get("robot_FSMs").items(),
                    blackboard.get("robot_types").values(),
                )
            ],
        )

    """
    Test fetch missions subtree
    """
    if test_fetch_mission_tree:
        FMTree = FleetManagerTree(3, test_graph, [Edison(), Edison(), EdyMobile()])
        FMTree.setup()
        fetch_missions_tree = create_fetch_missions_tree()
        behaviour_tree = py_trees.trees.BehaviourTree(fetch_missions_tree)
        print("TEST Fetch Missions Tree")
        behaviour_tree.tick()
        # Verify data is stored on the blackboard
        print("  Robot data", blackboard.get("robot_poses_goals"))

    """
    Test compute danger area
    """
    if test_compute_danger_area:
        # Add name, fsm, current pose
        robot_names = [
            "robot_test_DA_idle",
            "robot_test_DA_waiting",
            "robot_test_DA_move_sampling",
            "robot_test_DA_move_graph",
            "robot_test_DA_error",
        ]
        FMTree = FleetManagerTree(
            5,
            test_map,
            [EdyMobile(), EdyMobile(), EdyMobile(), EdyMobile(), EdyMobile()],
            robot_names,
        )
        FMTree.setup()
        fsms = blackboard.get("robot_FSMs")
        rpg = {}
        # node 26 at (3.0325, 7.87), 148 at (3.93, 8.16)
        # [26, 25, 27, 167, 168, 148] are connected
        rpg["robot_test_DA_idle"] = {"pose": (3.03, 7.8, 0), "goal": None}
        rpg["robot_test_DA_waiting"] = {"pose": (3.03, 7.8, 0), "goal": (3.93, 8.16, 0)}
        fsms["robot_test_DA_waiting"].startup()

        rrt_start = (3.03, 7.85, 0)
        rrt_goal = (3.93, 8.16, 0)
        rpg["robot_test_DA_move_sampling"] = {"pose": rrt_start, "goal": rrt_goal}

        rrt_path = RRT_star_Reeds_Shepp.create_and_plan(
            rrt_start, rrt_goal, 0.1, Edison(), test_map.map
        )
        robot_paths = {}
        if "robot_test_DA_move_sampling" not in robot_paths:
            robot_paths["robot_test_DA_move_sampling"] = {}
        robot_paths["robot_test_DA_move_sampling"]["pose_path"] = rrt_path
        blackboard.set("robot_paths", robot_paths)
        fsms["robot_test_DA_move_sampling"].startup()
        fsms["robot_test_DA_move_sampling"].resume_sampling()

        rpg["robot_test_DA_move_graph"] = {
            "pose": (3.03, 7.8, 0),
            "goal": (3.0325, 7.87, np.pi / 2),
        }
        if "robot_test_DA_move_graph" not in robot_paths:
            robot_paths["robot_test_DA_move_graph"] = {}
        node_path = [26, 25, 27, 167, 168, 148]
        robot_paths["robot_test_DA_move_graph"]["node_path"] = [
            26,
            25,
            27,
            167,
            168,
            148,
        ]
        geometry_path = []
        for i in range(len(node_path) - 1):
            geometry_path.append(
                test_graph.dynamic_graph.edges[(node_path[i], node_path[i + 1])][
                    "geometry"
                ]
            )
        robot_paths["robot_test_DA_move_graph"]["geometry_path"] = geometry_path
        blackboard.set("robot_paths", robot_paths)
        fsms["robot_test_DA_move_graph"].startup()
        fsms["robot_test_DA_move_graph"].resume_graph()

        rpg["robot_test_DA_error"] = {"pose": (3.03, 7.8, 0), "goal": (3.93, 8.16, 0)}
        blackboard.set("robot_poses_goals", rpg)
        fsms["robot_test_DA_error"].lose_control()

        blackboard.set("robot_paths", robot_paths)
        blackboard.set("robot_FSMs", fsms)

        compute_danger_area_tree = create_compute_danger_areas_tree(robot_names)
        behaviour_tree = py_trees.trees.BehaviourTree(compute_danger_area_tree)
        behaviour_tree.setup()
        behaviour_tree.tick()
        print("TEST Compute DA")
        danger_areas = blackboard.get("danger_areas")
        print("  Danger areas: ", danger_areas)
        if visualise_compute_danger_area:
            titles = ["IDLE", "WAITING", "MOVE SAMPLING", "MOVE GRAPH", "ERROR"]
            for danger_area, title in zip(danger_areas.values(), titles):
                plot_danger_area(test_map, danger_area, title)

    """
    Test block static obstacles
    """
    if test_block_static_obstacles:
        print("TEST block static obstacles")
        test_obstacle = Point(3.03, 7.85).buffer(0.1)
        test_graph.block_nodes_within_obstacle(test_obstacle)
        normal_path = nx.dijkstra_path(test_graph.full_graph, 166, 24)
        blocked_path = nx.dijkstra_path(test_graph.dynamic_graph, 166, 24)
        if normal_path != blocked_path:
            print("  Test successful. Path has changed due to obstacle")
        else:
            print("  Test failed. Path has not changed.")

    """
    Test path planning
    """
    if test_path_planning:
        robot_names = ["test_rrt_to_goal", "test_Astar_to_goal", "test_rrt_to_graph"]
        FMTree = FleetManagerTree(
            3,
            test_graph,
            [EdyMobile(), EdyMobile(), EdyMobile()],
            robot_names,
        )
        FMTree.setup()
        test_graph.reset_dynamic_graph()
        fsms = blackboard.get("robot_FSMs")
        rpg = {}
        # node 26 at (3.0325, 7.87), 148 at (3.93, 8.16)
        # [26, 25, 27, 167, 168, 148] are connected
        rpg["test_rrt_to_goal"] = {"pose": (3.03, 7.8, 0), "goal": (3.033, 7.87, np.pi / 2)}
        fsms["test_rrt_to_goal"].startup()
        rpg["test_Astar_to_goal"] = {
            "pose": (3.033, 7.87, np.pi / 2),
            "goal": (7.22, 15.45, 0),
        }
        fsms["test_Astar_to_goal"].startup()
        rpg["test_rrt_to_graph"] = {"pose": (7.2, 15.2, 0), "goal": (7.226, 15.6, -np.pi)}
        fsms["test_rrt_to_graph"].startup()

        blackboard.set("robot_poses_goals", rpg)
        print("TEST path planning")
        plan_paths_tree = create_plan_paths_tree(robot_names)
        behaviour_tree = py_trees.trees.BehaviourTree(plan_paths_tree)
        behaviour_tree.setup()
        behaviour_tree.tick()

        robot_paths = blackboard.get("robot_paths")
        for robot_name in robot_names:
            # node_path = robot_paths[robot_name]["node_path"]
            pose_path = robot_paths[robot_name]["pose_path"]
            # test_map.visualise(graph= test_graph.full_graph, path=node_path)
            # test_map.visualise(graph= test_graph.full_graph, pose_path=pose_path)
            if visualise_path_planning:
                test_map.visualise(pose_path=pose_path)
        print("  Test successful if no errors")

    """
    Test prioritise robots
    """
    if test_prioritise_robots:
        robot_names = [
            "robot_A",
            "robot_B"
        ]
        FMTree = FleetManagerTree(
            2,
            test_graph,
            [EdyMobile(), EdyMobile()],
            robot_names,
        )
        FMTree.setup()
        test_graph.reset_dynamic_graph()
        fsms = blackboard.get("robot_FSMs")
        fsms["robot_A"].startup()
        fsms["robot_B"].startup()
        blackboard.set("static_obstacles", [])
        """
        Test graph/graph
        """
        def get_pq_once(rpg):
            blackboard.set("robot_poses_goals", rpg)
            compute_danger_area_tree = create_compute_danger_areas_tree(robot_names)
            behaviour_tree = py_trees.trees.BehaviourTree(compute_danger_area_tree)
            behaviour_tree.setup()
            behaviour_tree.tick()
            plan_paths_tree = create_plan_paths_tree(robot_names)
            behaviour_tree = py_trees.trees.BehaviourTree(plan_paths_tree)
            behaviour_tree.setup()
            behaviour_tree.tick()
            prioritise_robot_tree = create_prioritise_and_move_robots_tree()
            behaviour_tree = py_trees.trees.BehaviourTree(prioritise_robot_tree)
            behaviour_tree.setup()
            behaviour_tree.tick()
            pq = blackboard.get("priority_list")
            return pq
        rpg_B_goes_first = {
            "robot_A": {
                "pose": (3.3575,4.26,np.pi/2),
                "goal": (3.3575,5.47,np.pi/2)
            },
            "robot_B": {
                "pose": (3.3575,4.28,np.pi/2),
                "goal": (3.3575,5.87,np.pi/2)
            }
        }
        rpg_A_goes_first = {
            "robot_A": {
                "pose": (3.3575,4.28,np.pi/2),
                "goal": (3.3575,5.87,np.pi/2)
            },
            "robot_B": {
                "pose": (3.3575,4.26,np.pi/2),
                "goal": (3.3575,5.47,np.pi/2)
            }
        }
        assert get_pq_once(rpg_B_goes_first) == ["robot_B", "robot_A"]
        assert get_pq_once(rpg_A_goes_first) == ["robot_A", "robot_B"]

        print("Test graph graph success")