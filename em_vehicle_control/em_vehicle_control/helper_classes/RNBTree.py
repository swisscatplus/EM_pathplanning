"""
Road Network Behaviour Tree
"""

from typing import Optional, List, Dict, Union, Tuple
import copy

import py_trees
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import substring
import networkx as nx

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
TICK_RATE = 8  # Hz
DANGER_AREA_BUFFER = 0.01  # m
DANGER_AREA_SWEEP_LOOKAHEAD = 2  # times of tick period
"""
Danger area sweep distance is given by 
DANGER_AREA_SWEEP_LOOKAHEAD / TICK_RATE * nominal speed of the robot
"""
RRT_GOAL_RADIUS = 0.1  # m
NEAR_TO_PATH_RADIUS = 0.05  # m
NEAR_TO_PATH_ANGLE = 0.8  # rad, high just to ensure direction is correct
NEAR_TO_GOAL_RADIUS = 0.05

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
"""


class CallForHelp(py_trees.behaviour.Behaviour):
    def __init__(self, name="Call for Help, Fix Error"):
        super().__init__(name)

    def update(self):
        print("Action: Calling for help, fixing error...")
        return py_trees.common.Status.SUCCESS


def find_nearest_point_on_geom_path(
    point: Point, path: List[LineString]
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


class GetAllCurrentPosesAndGoals(py_trees.behaviour.Behaviour):
    def __init__(self, name="Get All Current Poses and Goals"):
        super().__init__(name)

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
        blackboard.set("robot_FSMs", robot_fsms)

    def update(self) -> Status:
        # TODO: replace with ROS2 service
        robot_data = sim_get_poses_and_goals(2)
        for robot_name, _, _ in robot_data:
            assert robot_name in blackboard.get("robot_names")
        robot_poses_goals = {
            robot_name: {"pose": pose, "goal": goal}
            for robot_name, pose, goal in robot_data
        }
        blackboard.set("robot_poses_goals", robot_poses_goals)

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

    def register_static_obstacle(self, robot_polygon: Polygon) -> None:
        """
        Registers the robot as a static obstacle in the blackboard.

        Args:
            robot_polygon(Polygon): shape of robot to register
        """

        static_obstacles = blackboard.get("static_obstacles")
        if static_obstacles is None:
            static_obstacles = []

        static_obstacles.append(robot_polygon)
        blackboard.set("static_obstacles", static_obstacles)

    def block_graph_nodes(self, robot_polygon: Polygon) -> None:
        """
        Blocks all nodes, and their corresponding edges,
        on the dynamic_graph as inaccessible.

        Args:
            robot_polygon(Polygon): shape of robot to register
        """
        graph_network = blackboard.get("graph_network")
        blocked_nodes = graph_network.block_nodes_within_obstacle(robot_polygon)
        try:
            all_blocked_nodes = blackboard.get("blocked_nodes")
        except:
            all_blocked_nodes = []
        all_blocked_nodes += blocked_nodes
        blackboard.set("blocked_nodes", all_blocked_nodes)

    def update(self) -> Status:
        robot_fsms = blackboard.get("robot_FSMs")
        robot_poses_goals = blackboard.get("robot_poses_goals")
        robot_types = blackboard.get("robot_types")
        for robot_name, fsm in robot_fsms.items():
            if fsm.state() == "idle" or fsm.state() == "error":
                robot_pose = robot_poses_goals[robot_name]["pose"]
                robot_type = robot_types[robot_name]
                robot_type.construct_vehicle(robot_pose)
                robot_polygon = robot_type.vehicle_model.buffer(DANGER_AREA_BUFFER)
                self.register_static_obstacle(robot_polygon)
                self.block_graph_nodes(robot_polygon)
        return py_trees.common.Status.SUCCESS


class PlanPathForRobot(py_trees.behaviour.Behaviour):
    """
    Selects planning strategies and sets path for path tracker
    """

    def __init__(self, robot_name: str):
        super().__init__("Plan Path for Robot")
        self.robot_name = robot_name
        self.robot_type = None

    def plan_Astar(self, source: PosePt2D, goal: PosePt2D) -> List[int]:
        pass

    def plan_rrt_star(self, source: PosePt2D, goal: PosePt2D) -> List[PosePathPoint]:
        """
        Runs RRT* algorithm with Reeds Shepp pathing

        Args:
            source(PosePt2D): current pose of the robot
            goal(PosePt2D): desired pose of the robot

        Returns:
        """
        static_obstacles = blackboard.get("static_obstacles")

        pose_path = RRT_star_Reeds_Shepp.create_and_plan(
            source, goal, RRT_GOAL_RADIUS, self.robot_type, static_obstacles
        )

        return pose_path

    def find_nearest_point_on_posepoint_path(
        self, point: Point, path: List[PosePathPoint]
    ) -> Tuple[int, float]:
        """
        Identifies the closest point on the path to the lookup point.
        Used to look for closest point to path given by RRT* algorithm 
        as posepoints are close together, and interpolation is unnecessary.

        Args:
            point (Point): Lookup point
            path (List[PosePathPoint]): Path

        Returns:
            Tuple:
                - int: index of closest point on path to the lookup point
                - float: Distance between both points
        """
        nearest_distance_squared = np.inf
        nearest_pt_idx = None
        for idx, pose_point in enumerate(path):
            dist_squared = (point.x - pose_point[0]) ** 2 + (
                point.y - pose_point[1]
            ) ** 2
            if not dist_squared < nearest_distance_squared**2:
                continue
            nearest_distance_squared = dist_squared
            nearest_pt_idx = idx
        return nearest_pt_idx, np.sqrt(nearest_distance_squared)

    def get_angle_at_posepoint_path(self, path: List[PosePathPoint], index: int) -> float:
        """
        Gets the angle of a point on the path with respect to the world frame

        Args:
            path (List[PosePathPoint]): Path
            index (int): Index of point to measure angle around

        Returns:
            float: measured angle
        """
        # ensure front and back points are on the same direction path
        # use central difference formula if possible
        if (
            index == 0 or
            path[index][2] != path[index-1][2]
        ):
            back_point = path[index]
        else:
            back_point = path[index-1]
        if (
            index == len(path) - 1 or
            path[index][2] != path[index+1][2]
        ):
            front_point = path[index]
        else:
            front_point = path[index+1]
        
        # if fail to find 2 different points with above methd
        split = 1
        while front_point == back_point:
            back_point = path[np.max(index-split, 0)]
            front_point = path[np.min(index+split, len(path)-1)]
            split += 1
        angle = np.arctan2(front_point[1]-back_point[1], front_point[0]-back_point[0])
        if path[index][2] == -1:
            # direction is backwards, flip pi rad
            angle = angle + np.pi
            angle = (angle + np.pi) % (2 * np.pi) - np.pi # mod 2pi
        return angle

    def near_to_posepoint_path(self, current_pose: PosePt2D, path: List[PosePathPoint]) -> bool:
        """
        Checks if robot is near to a pose point path.
        Uses global var NEAR_TO_PATH_RADIUS and NEAR_TO_PATH_ANGLE (Is this necessary?)

        Args:
            current_pose(PosePt2D): robot coordinates (x,y, yaw)
            path(List[PosePathPoint]): pose path

        Returns:
            bool: true if near to path
        """
        current_pose_pt = Point(current_pose[0], current_pose[1])
        closest_pt_idx, shortest_distance = self.find_nearest_point_on_posepoint_path(current_pose_pt, path)
        angle = self.get_angle_at_posepoint_path(path, closest_pt_idx)
        if (
            shortest_distance < NEAR_TO_PATH_RADIUS and
            ((angle - current_pose[2] + np.pi) % 2*np.pi - np.pi) < NEAR_TO_PATH_ANGLE
        ):
            return True
        return False

    def near_to_goal(self, current_pose: PosePt2D, goal: PosePt2D) -> bool:
        """
        Checks if the robot is near to the goal, to switch path planning.
        Not for checking if the robot is at the goal to switch to idle state!

        Args:
            current_pose(PosePt2D): robot coordinates (x,y, yaw)
            goal(PosePt2D): goal location
        
        Returns:
            bool: true if near to goal
        """
        dist = np.sqrt((current_pose[0] - goal[0])**2 + (current_pose[1] - goal[1])**2)
        if dist < NEAR_TO_GOAL_RADIUS:
            return True
        return False

    def near_to_geom_path(self, current_pose: PosePt2D, path: List[LineString]) -> bool:
        """
        Checks if robot is near to a graph edge path.
        Uses global var NEAR_TO_PATH_RADIUS

        Args:
            current_pose(PosePt2D): robot coordinates (x,y, yaw)
            path(List[LineString]): geom path

        Returns:
            bool: true if near to path
        """
        pose = Point(current_pose[0], current_pose[1])
        nearest_point, _ = find_nearest_point_on_geom_path(pose, path)
        dist = np.sqrt((current_pose[0] - nearest_point.x)**2 + (current_pose[1] - nearest_point.y)**2)
        if dist < NEAR_TO_PATH_RADIUS:
            return True
        return False

    def near_to_graph_node(self, current_pose: PosePt2D, goal: PosePt2D, graph_network: RoadTrack) -> Tuple[bool, PosePt2D]:
        """
        Checks if robot is near to a node.
        Uses global var NEAR_TO_PATH_RADIUS

        Args:
            current_pose(PosePt2D): robot coordinates (x,y, yaw)
            graph_network(RoadTrack): RoadTrack object that contains all nodes
        
        Returns:
            Tuple:
                - bool: True if near to node
                - PosePt2D: (x,y,yaw) of nearest node, yaw from running A* on current pose to goal
        """
        graph_network

    def update(self) -> Status:
        robot_poses_goals = blackboard.get("robot_poses_goals")
        robot_pose_goal = robot_poses_goals[self.robot_name]
        robot_pose = robot_pose_goal["pose"]
        robot_goal = robot_pose_goal["goal"]
        robot_fsms = blackboard.get("robot_FSMs")
        robot_fsm = robot_fsms[self.robot_name]
        robot_state = robot_fsm.state
        robot_paths = blackboard.get("robot_paths")
        robot_paths = robot_paths[self.robot_name]
        pose_path = robot_paths.get("pose_path")
        geometry_path = robot_paths.get("geometry_path")
        node_path = robot_paths.get("node_path")
        graph_network = blackboard.get("graph_network")
        if (
            robot_state == "move_by_sampling"
            and pose_path is not None
            and self.near_to_posepoint_path(robot_pose, pose_path)
        ):
            """
            in move_by_sampling state
            no change in path, continue on pose_path
            """
            return py_trees.common.Status.SUCCESS
        elif self.near_to_goal(robot_pose):
            """
            In waiting, move_by_sampling, or move_by_graph state
            near to goal, move to goal
            """
            new_pose_path = self.plan_rrt_star(robot_pose, robot_goal)

            # TODO process path here
            return py_trees.common.Status.SUCCESS
        elif (
            robot_state == "move_by_graph"
            and node_path is not None
            and self.near_to_geom_path(robot_pose, geometry_path)
        ):
            """
            in move_by_graph state,
            no change in path, continue on pose_path
            """
            return py_trees.common.Status.SUCCESS
        elif robot_state == "waiting" and self.near_to_graph_node(robot_pose):
            """
            in waiting state,
            check if ready for graph based planing
            """
            new_node_path = self.plan_Astar(robot_pose, robot_goal)
            # TODO process path here
            return py_trees.common.Status.SUCCESS
        elif robot_state == "waiting":
            new_robot_goal = self.get_graph_node_goal(robot_pose)
            new_pose_path = self.plan_rrt_star(robot_pose, new_robot_goal)

            # TODO process path here
            return py_trees.common.Status.SUCCESS

        # unknown state of system, returns failure
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
    # TODO
    # Add a behavior for handling all idle, waiting, and error robots
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

            if remaining_distance <= 0:
                # Clip the final segment to fit the exact remaining distance
                excess_length = abs(remaining_distance)
                final_segment = substring(
                    segment_part, 0, segment_part.length - excess_length
                )
                upcoming_coords = upcoming_coords[: -(len(segment_part.coords))]
                upcoming_coords.extend(final_segment.coords)
                break

        return LineString(upcoming_coords)

    def update(self) -> py_trees.common.Status:
        robot_pose = blackboard.get("robot_poses_goals")[self.robot_name]["pose"]
        self.robot_type.construct_vehicle(robot_pose)
        robot_footprint = self.robot_type.vehicle_model
        robot_fsm = blackboard.get("robot_FSMs")[self.robot_name]
        if not robot_fsm:
            print(f"FSM for robot {self.robot_id} not found on blackboard.")
            return py_trees.common.Status.FAILURE

        danger_area = None
        if robot_fsm.state == "error" or robot_fsm.state == "waiting":
            danger_area = robot_footprint.buffer(DANGER_AREA_BUFFER)
        elif robot_fsm.state == "idle":
            danger_area = robot_footprint.buffer(DANGER_AREA_BUFFER)
        elif robot_fsm.state == "move_by_sampling":
            # TODO confirm if RRT* path is saved as such
            path = copy.deepcopy(
                blackboard.get("robot_paths")[self.robot_name]["pose_path"]
            )
            if not path:
                "Compute danger area error, no path planned while in state move_sampling"
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
                "Compute danger area error, no path planned while in state move_graph"
                return py_trees.common.Status.FAILURE
            current_pose = Point(robot_pose[0], robot_pose[1])
            nearest_point, closest_seg_idx = find_nearest_point_on_geom_path(
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


def create_prioritise_and_move_robots_tree():
    root = py_trees.composites.Sequence(
        "Prioritise and Move Robots Subtree", memory=False
    )
    # TODO
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

        self.root = self.create_fleet_manager_tree()
        self.tree = py_trees.trees.BehaviourTree(self.root)

    def create_fleet_manager_tree(self):
        root = py_trees.composites.Selector("Fleet Manager", memory=False)

        call_for_help = CallForHelp()

        fetch_missions_tree = create_fetch_missions_tree()
        plan_paths_tree = create_plan_paths_tree(self.robot_names)
        compute_danger_areas_tree = create_compute_danger_areas_tree(self.robot_names)
        prioritise_and_move_robots_tree = create_prioritise_and_move_robots_tree()

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

        root.add_children([call_for_help, main_tasks_sequence])
        return root

    def tick(self):
        self.tree.tick()

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
        FMTree = FleetManagerTree(3, test_map, [Edison(), Edison(), EdyMobile()])
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
        FMTree = FleetManagerTree(3, test_map, [Edison(), Edison(), EdyMobile()])
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
                test_graph.full_graph.edges[(node_path[i], node_path[i + 1])][
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

        compute_danger_area_tree = create_compute_danger_areas_tree(5, robot_names)
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
    test_obstacle = Point(3.03, 7.85).buffer(0.1)
    test_graph.block_nodes_within_obstacle(test_obstacle)
    normal_path = nx.dijkstra_path(test_graph.full_graph, 166, 24)
    blocked_path = nx.dijkstra_path(test_graph.dynamic_graph, 166, 24)
