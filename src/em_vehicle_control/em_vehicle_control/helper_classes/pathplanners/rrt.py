from copy import deepcopy
import numpy as np
from typing import Tuple, Union, Optional, List
from shapely import Polygon, Point, LineString, overlaps, within
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from em_vehicle_control.helper_classes.vehicles import *
from em_vehicle_control.helper_classes.map import RoadMap
from em_vehicle_control.helper_classes.pathplanners.reeds_shepp_path_planning import (
    reeds_shepp_path_planning,
)


class RRTNode:
    """
    Nodes for RRT and RRT_star
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

    def get_pose(self):
        """
        return: pose
        """
        return self.x, self.y


class RRT:
    """
    Rapidly random trees
    """

    def __init__(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        goal_radius: float,
        vehicle: Union[EdyMobile],
        local_map: Polygon,
        other_vehicles: Optional[Polygon] = [],
        collision_buffer: float = 0.05,
        step_size: float = 0.1,
        max_iter: int = 100,
        seed: int = 42,
        visualise: bool = False,
    ):
        """
        start: (x,y) coordinates of vehicle
        goal: (x, y) coordinates of goal position
        goal_radius: acceptable region to be in the goal
        vehicle: that has its path be planned
        local_map: map of the local region around current vehicle
        other_vehicles: any other vehicles in the local_map
        collision_buffer: size around vehicle that counts as a collision
        step_size: step size to move toward the random point
        max_iter: maximum number of random points to sample
        seed: for random
        visualise: visualises algo
        """
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.vehicle = vehicle
        self.local_map = local_map
        self.other_vehicles = other_vehicles
        self.collision_buffer = collision_buffer
        self.step_size = step_size
        self.max_iter = max_iter
        self.visualise = visualise
        np.random.seed(seed)

        start_node = RRTNode(start[0], start[1])
        start_node.parent = -1
        self.node_list = [start_node]
        self.path = []

        # gets map size
        self.minx, self.miny, self.maxx, self.maxy = self.local_map.bounds

        if self.visualise:
            self.fig, self.ax = plt.subplots()
            self.frames = []

    def sample_point(self) -> Tuple[float, float]:
        """
        return: point sampled
        """
        x_rand = np.random.rand() * (self.maxx - self.minx) + self.minx
        y_rand = np.random.rand() * (self.maxy - self.miny) + self.miny
        return x_rand, y_rand

    def steer(self, from_node: RRTNode, sampled_point: Tuple[float, float]) -> RRTNode:
        """
        from_node: node to steer from
        sampled_point: direction to steer the node in
        return: a node in the direction of the sampled point from the from_node
        """
        _, angle = self.calc_distance_and_angle(from_node.get_pose(), sampled_point)
        candidate_x = from_node.x + self.step_size * np.cos(angle)
        candidate_y = from_node.y + self.step_size * np.sin(angle)
        candidate_node = RRTNode(candidate_x, candidate_y)
        return candidate_node

    def nearest_neighbor_idx(
        self, sampled_pt: Union[Tuple[float, float], Tuple[float, float, float]]
    ):
        """
        sampled_pt: position of a sampled point
        return: index of nearest node to sampled point
        """
        distances = [
            (node.x - sampled_pt[0]) ** 2 + (node.y - sampled_pt[1]) ** 2
            for node in self.node_list
        ]
        return distances.index(min(distances))

    def check_collision_point(self, pose: Tuple[float, float, float]) -> bool:
        """
        1. Check that our vehicle is completely inside the map
        2. Check that our vehicle will not collide with any others
        point: the vehicle at this pose will be check if there is a collision in the map
        return: true if there is a collision
        """
        temp_vehicle = deepcopy(self.vehicle)
        temp_vehicle._x = pose[0]
        temp_vehicle._y = pose[1]
        temp_vehicle._theta = pose[2]
        temp_vehicle.construct_vehicle()
        if not within(temp_vehicle._vehicle_model, self.local_map):
            return True
        for other_vehicle in self.other_vehicles:
            if overlaps(other_vehicle, temp_vehicle._vehicle_model):
                return True
        return False

    def check_collision_line(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        resolution: Optional[float] = 0.1,
    ) -> bool:
        """
        start_point: a line is defined starting from here
        end_point: to here
        resolution: The vehicle will be checked along the line at a spacing defined here in metres
        return: true if there is a collision
        """
        dist, theta = self.calc_distance_and_angle(start_point, end_point)
        N = np.ceil(dist / resolution).astype(int)
        step = dist / N
        for i in range(N):
            check_x = start_point[0] + step * i * np.cos(theta)
            check_y = start_point[1] + step * i * np.sin(theta)
            if self.check_collision_point((check_x, check_y, theta)):
                return True
        return self.check_collision_point((*end_point, theta))

    def check_in_goal(self, node: RRTNode) -> bool:
        """
        node: to check
        returns: True if node is within goal radius distance of goal
        """
        dist_to_goal, _ = self.calc_distance_and_angle(node.get_pose(), self.goal)
        return dist_to_goal < self.goal_radius

    def get_path(self) -> List[int]:
        """
        returns: path from start of the node list to the end
        """
        reversed_path = [len(self.node_list) - 1]
        prev_node_idx = self.node_list[-1].parent
        while prev_node_idx != -1:
            reversed_path.append(prev_node_idx)
            prev_node_idx = self.node_list[prev_node_idx].parent
        return list(reversed(reversed_path))

    def get_path_from(self, end_idx: int) -> List[int]:
        """
        end_idx: index of the last node of the path
        returns: path from start of the node list to the end index
        """
        reversed_path = [end_idx]
        prev_node_idx = self.node_list[end_idx].parent
        while prev_node_idx != -1:
            reversed_path.append(prev_node_idx)
            prev_node_idx = self.node_list[prev_node_idx].parent
        return list(reversed(reversed_path))

    @staticmethod
    def calc_distance_and_angle(start_point, end_point):
        """
        start_point: (x, y)
        end_point: (x, y)
        return: distance and angle
        """
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        d = np.hypot(dx, dy).astype(float)
        theta = np.arctan2(dy, dx).astype(float)
        return d, theta

    def plot_rrt(self) -> None:
        """
        Plots rrt graph and returns list of artists for animation
        """
        self.ax.clear()
        artists = []  # Initialize an empty list for artists

        patch, _ = RoadMap.shapely_to_mplpolygons(self.vehicle._vehicle_model)
        self.ax.add_patch(patch)
        artists.append(patch)
        for other_vehicle in self.other_vehicles:
            patch, _ = RoadMap.shapely_to_mplpolygons(other_vehicle)
            self.ax.add_patch(patch)
            artists.append(patch)
        patch, interior_polygons = RoadMap.shapely_to_mplpolygons(self.local_map)
        self.ax.add_patch(patch)
        artists.append(patch)
        for patch in interior_polygons:
            self.ax.add_patch(patch)
            artists.append(patch)

        # Plot the RRT nodes and edges
        for node in self.node_list:
            if node.parent != -1:
                parent_node = self.node_list[node.parent]
                line = self.ax.plot(
                    [node.x, parent_node.x], [node.y, parent_node.y], "b-"
                )
                artists.extend(line)  # Collect the line artist(s)

        # Plot start and goal points
        start_artist = self.ax.plot(self.start[0], self.start[1], "go", markersize=10)
        goal_artist = self.ax.plot(self.goal[0], self.goal[1], "ro", markersize=10)
        artists.extend(start_artist)  # Collect the start point artist
        artists.extend(goal_artist)  # Collect the goal point artist

        # Plot the final path if it exists
        if self.path != []:
            path_coords = np.array(
                [[self.node_list[i].x, self.node_list[i].y] for i in self.path]
            )
            path_artist = self.ax.plot(
                path_coords[:, 0], path_coords[:, 1], "r-", linewidth=2
            )
            artists.extend(path_artist)  # Collect the path artist

        self.ax.axis("scaled")
        return artists  # Return the list of artists

    def plan(self) -> None:
        """
        Main iterating algorithm of RRT
        return: True if success
        """
        success = False
        for _ in range(self.max_iter):
            sampled_point = self.sample_point()
            nearest_node_idx = self.nearest_neighbor_idx(sampled_point)
            nearest_node = self.node_list[nearest_node_idx]
            candidate_node = self.steer(nearest_node, sampled_point)
            if self.check_collision_line(
                nearest_node.get_pose(), candidate_node.get_pose()
            ):
                # collide! add debug messages maybe
                continue
            candidate_node.parent = nearest_node_idx
            self.node_list.append(candidate_node)
            if self.check_in_goal(candidate_node):
                # Success!
                self.path = self.get_path()
                success = True
                break
        if self.visualise:
            self.plot_rrt()
            plt.show()
        return success


class RRT_star(RRT):
    """
    RRT star by Frazzoli and Karaman,
    includes keeping a cost, and rewiring
    """

    def __init__(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        goal_radius: float,
        vehicle: Union[EdyMobile],
        local_map: Polygon,
        other_vehicles: Optional[List[Union[EdyMobile]]] = [],
        collision_buffer: float = 0.05,
        step_size: float = 0.1,
        max_iter: int = 100,
        ball_beta: float = 20.0,
        search_until_max_iter: bool = False,
        seed: int = 42,
        visualise: bool = False,
    ):
        """
        start: (x,y) coordinates of vehicle
        goal: (x, y) coordinates of goal position
        goal_radius: acceptable region to be in the goal
        vehicle: that has its path be planned
        local_map: map of the local region around current vehicle
        other_vehicles: any other vehicles in the local_map
        collision_buffer: size around vehicle that counts as a collision
        step_size: step size to move toward the random point
        max_iter: maximum number of random points to sample
        ball_beta: neighbouring nodes will be searched in ball_beta*log(n)/n region
        search_until_max_iter: early stopping or otherwise
        seed: for random
        visualise: visualises algo
        """
        super().__init__(
            start,
            goal,
            goal_radius,
            vehicle,
            local_map,
            other_vehicles,
            collision_buffer,
            step_size,
            max_iter,
            seed,
            visualise,
        )
        self.ball_beta = ball_beta
        self.search_until_max_iter = search_until_max_iter
        self.goal_node = RRTNode(goal[0], goal[1])
        self.node_list: List[RRTNode] = []

    def get_nearby_nodes(self, candidate_node: RRTNode) -> List[int]:
        """
        candidate_node: node to search around
        return: list of indices of nodes that are inside the ball
        ball radius is defined as beta_ball * log(n) / n
        """
        distances = [
            (node.x - candidate_node.x) ** 2 + (node.y - candidate_node.y) ** 2
            for node in self.node_list
        ]
        n = len(self.node_list)
        ball_radius = (self.ball_beta * np.log(n) / n) ** 2
        nearby_nodes = [distances.index(i) for i in distances if i < ball_radius]
        if nearby_nodes == []:
            nearby_nodes = [np.argmin(distances)]
        return nearby_nodes

    def choose_parent(
        self, candidate_node: RRTNode, nearby_idxs: List[int]
    ) -> Tuple[RRTNode, bool]:
        """
        candidate_node: new node
        nearby_indx: list of all possible parent nodes
        returns: updated new node and whether it was changed or not
        Searches through list of parents, and returns the parent with the lowest cost to the ball
        """
        changed = False
        for nearby_idx in nearby_idxs:
            nearby_node = self.node_list[nearby_idx]
            added_cost, _ = RRT.calc_distance_and_angle(
                nearby_node.get_pose(), candidate_node.get_pose()
            )
            if (
                added_cost + nearby_node.cost < candidate_node.cost
                and not self.check_collision_line(
                    nearby_node.get_pose(), candidate_node.get_pose()
                )
            ):
                changed = True
                candidate_node.cost = added_cost + nearby_node.cost
                candidate_node.parent = nearby_idx
        return candidate_node, changed

    def rewire(self, candidate_node: RRTNode, nearby_idxs: List[int]) -> None:
        """
        For all nearby nodes in the ball,
        """
        for nearby_idx in nearby_idxs:
            nearby_node = self.node_list[nearby_idx]
            added_cost, _ = RRT.calc_distance_and_angle(
                candidate_node.get_pose(), nearby_node.get_pose()
            )
            if (
                added_cost + candidate_node.cost < nearby_node.cost
                and not self.check_collision_line(
                    candidate_node.get_pose(), nearby_node.get_pose()
                )
            ):
                nearby_node.cost = added_cost + candidate_node.cost
                nearby_node.parent = len(
                    self.node_list
                )  # since we know the candidate node is now appended
                self.node_list[nearby_idx] = nearby_node

    def best_node_to_goal(self) -> Union[int, None]:
        """
        return: index corresponding to the node closest to the goal
        """
        distances = [
            (node.x - self.goal[0]) ** 2 + (node.y - self.goal[1]) ** 2
            for node in self.node_list
        ]
        if min(distances) > self.goal_radius**2:
            return None
        return distances.index(min(distances))

    def plan(self) -> None:
        """
        RRT star path planning
        """
        start_node = RRTNode(*self.start)
        start_node.parent = -1
        self.node_list.append(start_node)
        success = False
        for _ in range(self.max_iter):
            sampled_point = self.sample_point()
            nearest_node_idx = self.nearest_neighbor_idx(sampled_point)
            nearest_node = self.node_list[nearest_node_idx]
            candidate_node = self.steer(nearest_node, sampled_point)
            added_cost, _ = RRT.calc_distance_and_angle(
                nearest_node.get_pose(), candidate_node.get_pose()
            )
            candidate_node.cost = nearest_node.cost + added_cost
            candidate_node.parent = nearest_node_idx
            if self.check_collision_line(
                nearest_node.get_pose(), candidate_node.get_pose()
            ):
                continue
            nearby_nodes_idx = self.get_nearby_nodes(candidate_node)
            candidate_node, changed = self.choose_parent(
                candidate_node, nearby_nodes_idx
            )
            if changed:
                self.rewire(candidate_node, nearby_nodes_idx)
            self.node_list.append(candidate_node)
            if not self.search_until_max_iter:
                if self.check_in_goal(candidate_node):
                    # Success!
                    self.path = self.get_path()
                    success = True
                    break
        if self.search_until_max_iter:
            end_node_idx = self.best_node_to_goal()
            if end_node_idx is not None:
                self.path = self.get_path_from(end_node_idx)
        if self.visualise:
            self.plot_rrt()
            plt.show()
        return success


class RRT_star_Reeds_Shepp_Node(RRTNode):
    """
    Nodes with yaw, for reeds shepp pathing
    """

    def __init__(self, x, y, yaw):
        super().__init__(x, y)
        self.yaw = yaw
        self.px = []
        self.py = []
        self.pyaw = []
        self.pdir = [] # -1 reverse, 1 forward

    def get_pose(self):
        """
        return x, y, yaw of node
        """
        return (self.x, self.y, self.yaw)
    
    def __repr__(self) -> str:
        return f"Node at {self.x:.2f}, {self.y:.2f}, {self.yaw:.2f}"


class RRT_star_Reeds_Shepp(RRT_star):
    """
    RRT star with reeds shepp steering
    """

    def __init__(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        goal_radius: float,
        vehicle: Union[EdyMobile, Edison],
        local_map: Polygon,
        other_vehicles: Optional[Polygon] | None = [],
        collision_buffer: float = 0.05,
        step_size: float = 0.1,
        max_iter: int = 100,
        ball_beta: float = 20,
        search_until_max_iter: bool = False,
        seed: int = 42,
        visualise: bool = False,
    ):
        super().__init__(
            start,
            goal,
            goal_radius,
            vehicle,
            local_map,
            other_vehicles,
            collision_buffer,
            step_size,
            max_iter,
            ball_beta,
            search_until_max_iter,
            seed,
            visualise,
        )
        self.node_list: List[RRT_star_Reeds_Shepp_Node] = []
        self.goal_node = RRT_star_Reeds_Shepp_Node(*goal)
        start_node = RRT_star_Reeds_Shepp_Node(*start)
        start_node.parent = -1
        self.node_list = [start_node]
        self.path = None
    
    @classmethod
    def create_and_plan(cls, *args, **kwargs):
        """
        Factory method to create an instance and return the planned path.
        """
        instance = cls(*args, **kwargs)
        if instance.plan():
            return instance.return_path()
        else: 
            return None

    def sample_point(self) -> Tuple[float, float, float]:
        """
        return: point sampled
        """
        x_rand = np.random.rand() * (self.maxx - self.minx) + self.minx
        y_rand = np.random.rand() * (self.maxy - self.miny) + self.miny
        yaw_rand = np.random.rand() * 2 * np.pi - np.pi

        return (x_rand, y_rand, yaw_rand)
    
    def pick_candidate_node(self, pose: Tuple[float, float, float], from_node: RRT_star_Reeds_Shepp_Node) -> RRT_star_Reeds_Shepp_Node:
        """
        pose: (x,y,yaw)
        nearest_node: node nearest to pose
        return: node at step_size away from the nearest_node
        """
        from_x, from_y, _ = from_node.get_pose()
        pose_x, pose_y, pose_yaw = pose
        _, angle = self.calc_distance_and_angle((from_x, from_y), (pose_x, pose_y))
        candidate_x = from_node.x + self.step_size * np.cos(angle)
        candidate_y = from_node.y + self.step_size * np.sin(angle)
        return RRT_star_Reeds_Shepp_Node(candidate_x, candidate_y, pose_yaw)
    
    # def nearest_neighbor_idx(
    #     self, sampled_pt: RRT_star_Reeds_Shepp_Node
    # ):
    #     """
    #     sampled_pt: position of a sampled point
    #     return: index of nearest node to sampled point
    #     """
    #     distances = [
    #         (node.x - sampled_pt.x) ** 2 + (node.y - sampled_pt.y) ** 2
    #         for node in self.node_list
    #     ]
    #     return distances.index(min(distances))

    def steer(
        self,
        from_node: RRT_star_Reeds_Shepp_Node,
        candidate_node: RRT_star_Reeds_Shepp_Node,
    ) -> RRT_star_Reeds_Shepp_Node:
        """
        from_node: node to steer from
        sampled_point: direction to steer the node in
        return: a node in the direction of the sampled point from the from_node
        """
        px, py, pyaw, pdir, _, length = reeds_shepp_path_planning(
            *from_node.get_pose(),
            *candidate_node.get_pose(),
            maxc=1/self.vehicle.turning_radius,
            step_size=0.02
        )
        if not px:
            return None

        candidate_node.px = px
        candidate_node.py = py
        candidate_node.pyaw = pyaw
        candidate_node.pdir = pdir
        candidate_node.cost += sum([abs(l) for l in length])
        
        return candidate_node
    
    def check_collision_linestring(self, node: RRT_star_Reeds_Shepp_Node) -> bool:
        """
        Checks path defined in node for collisions
        node: candidate node with path information
        return: true if there is a collision
        """
        for x, y, yaw in zip(node.px, node.py, node.pyaw):
            if self.check_collision_point((x, y, yaw)):
                return True
        return False
    
    def rewire(self, candidate_node: RRT_star_Reeds_Shepp_Node, nearby_idxs: List[int]) -> RRT_star_Reeds_Shepp_Node:
        """
        candidate_node: new node
        nearby_indx: list of all possible parent nodes
        returns: updated new node and whether it was changed or not
        Searches through list of parents, and returns the parent with the lowest cost to the ball
        """
        for nearby_idx in nearby_idxs:
            nearby_node = self.node_list[nearby_idx]
            px, py, pyaw, pdir, _, length = reeds_shepp_path_planning(
                *nearby_node.get_pose(),
                *candidate_node.get_pose(),
                maxc=1/self.vehicle.turning_radius,
                step_size=0.005
            )
            if not px:
                continue
            temp_node = deepcopy(candidate_node)
            temp_node.px = px
            temp_node.py = py
            temp_node.pyaw = pyaw
            temp_node.pdir = pdir
            added_cost = sum([abs(l) for l in length])
            if(
                added_cost + nearby_node.cost < candidate_node.cost
                and not self.check_collision_linestring(temp_node)
            ):
                candidate_node = deepcopy(temp_node)
                candidate_node.parent = nearby_idx
        return candidate_node
    
    def try_goal_path(self, node: RRT_star_Reeds_Shepp_Node) -> bool:
        """
        node: possible node to path towards the goal
        return: true if path to goal found
        """
        px, py, pyaw, pdir, _, length = reeds_shepp_path_planning(
            *node.get_pose(),
            *self.goal_node.get_pose(),
            maxc=1/self.vehicle.turning_radius,
            step_size=0.005
        )
        if not px:
            return False
        added_cost = sum([abs(l) for l in length])
        temp_node = RRT_star_Reeds_Shepp_Node(*self.goal_node.get_pose())
        temp_node.px = px
        temp_node.py = py
        temp_node.pyaw = pyaw
        temp_node.pdir = pdir
        if self.check_collision_linestring(temp_node):
            return False
        if self.goal_node.cost != 0:
            # that means a previous goal route has been found
            if not added_cost + node.cost < self.goal_node.cost:
                return False
        self.goal_node.cost = added_cost + node.cost
        self.goal_node.px = px
        self.goal_node.py = py
        self.goal_node.pyaw = pyaw
        self.goal_node.pdir = pdir
        return True
    
    def get_path(self) -> List[int]:
        """
        return: list of indices corresponding to the found RRT* path
        """
        self.node_list.append(self.goal_node)
        return super().get_path()

    def plan(self) -> bool:
        """
        RRT star planning with reeds shepp steering
        """

        success = False
        goal_found = False
        for i in range(self.max_iter):
            sampled_point = self.sample_point()
            nearest_node_idx = self.nearest_neighbor_idx(sampled_point)
            nearest_node = self.node_list[nearest_node_idx]
            candidate_node = self.pick_candidate_node(sampled_point, nearest_node)
            if self.check_collision_point(candidate_node.get_pose()):
                continue
            candidate_node = self.steer(nearest_node, candidate_node)
            if candidate_node is None:
                continue
            candidate_node.parent = nearest_node_idx
            if self.check_collision_linestring(candidate_node):
                continue
            # rewire here
            nearby_nodes_idxs = self.get_nearby_nodes(candidate_node)
            candidate_node = self.rewire(candidate_node, nearby_nodes_idxs)
            self.node_list.append(candidate_node)
            if self.try_goal_path(candidate_node):
                goal_found = True
                self.goal_node.parent = len(self.node_list) - 1 # candidate node index
            if not self.search_until_max_iter and goal_found:
                self.path = self.get_path()
                success = True
                break
        if self.search_until_max_iter and goal_found:
            self.path = self.get_path()
            success = True
        if self.visualise:
            self.plot_rrt()
            plt.show()
        if not success:
            print("Warning: RRT star with reeds-shepp path failed!")
        return success
    
    def return_path(self) -> List[Tuple[float, float, int]]:
        geometric_path_x = []
        geometric_path_y = []
        geometric_path_dir = []
        for node in self.path:
            geometric_path_x += (self.node_list[node].px)
            geometric_path_y += (self.node_list[node].py)
            geometric_path_dir += (self.node_list[node].pdir)
        return list(zip(geometric_path_x, geometric_path_y, geometric_path_dir))
    
    def plot_rrt(self) -> None:
        """
        Plots rrt graph and returns list of artists for animation
        """
        self.ax.clear()
        artists = []  # Initialize an empty list for artists

        patch, _ = RoadMap.shapely_to_mplpolygons(self.vehicle._vehicle_model)
        self.ax.add_patch(patch)
        artists.append(patch)
        for other_vehicle in self.other_vehicles:
            patch, _ = RoadMap.shapely_to_mplpolygons(other_vehicle)
            self.ax.add_patch(patch)
            artists.append(patch)
        patch, interior_polygons = RoadMap.shapely_to_mplpolygons(self.local_map)
        self.ax.add_patch(patch)
        artists.append(patch)
        for patch in interior_polygons:
            self.ax.add_patch(patch)
            artists.append(patch)

        # Plot the RRT nodes and edges
        for node in self.node_list:
            if node.parent != -1:
                parent_node = self.node_list[node.parent]
                line = self.ax.plot(node.px, node.py, "b-")
                artists.extend(line)  # Collect the line artist(s)

        # Plot start and goal points
        start_artist = self.ax.plot(self.start[0], self.start[1], "go", markersize=10)
        goal_artist = self.ax.plot(self.goal[0], self.goal[1], "ro", markersize=10)
        artists.extend(start_artist)  # Collect the start point artist
        artists.extend(goal_artist)  # Collect the goal point artist

        # Plot the final path if it exists
        if self.path != []:
            path_px = []
            path_py = []

            # Process the nodes in the path from start to goal
            for index in self.path:
                node = self.node_list[index]
                if node.parent != -1:
                    # Append the path from parent to node
                    path_px.extend(node.px)
                    path_py.extend(node.py)
                else:
                    # For the start node, just use its position
                    path_px.append(node.x)
                    path_py.append(node.y)

            path_artist = self.ax.plot(
                path_px, path_py, "r-", linewidth=2
            )
            artists.extend(path_artist)

        self.ax.axis("scaled")
        return artists  # Return the list of artists
    
if __name__ == '__main__':
    from helper_classes.map import RoadMap, RoadSegment, RoadGraph, RoadTrack
    from helper_classes.global_plannar import GlobalPlannar
    from helper_classes.vehicles import EdyMobile, Edison
    from helper_classes.pathplanners.rrt import *
    from helper_classes.pathplanners.pp_viz import RoadMapAnimator

    test_roads = [
        RoadSegment((2.87,1.67), (3.52,4.67)),
    ]
    test_map = RoadMap(test_roads)
    vehicle = EdyMobile(start_position=(3.218,3,np.pi/2))
    goal = (3.218,3,0)
    goal_radius = 0.2
    rrt = RRT_star_Reeds_Shepp((3.218,3,np.pi/2), goal, goal_radius, vehicle, test_map.map, 
                max_iter=2000, search_until_max_iter=False, visualise=True)