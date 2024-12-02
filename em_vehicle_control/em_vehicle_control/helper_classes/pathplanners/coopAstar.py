from typing import List, Optional, Union, Tuple, Dict, Set, TypeVar
from dataclasses import dataclass
import heapq
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely import buffer

from ..map import RoadGraph, RoadMap
from ..vehicles import *

VehicleType = TypeVar("VehicleType", EdyMobile, Edison)
ReservationEntry = Tuple[Polygon, float, float]
ReservationTable = List[ReservationEntry]


@dataclass
class TimedNode:
    """Represents a node with associated timing information."""

    index: int  # Node index
    enter_time: float  # Time entering the node
    exit_time: float  # Time exiting the node


class CAstarNode:
    """Node class for Cooperative A* algorithm."""

    def __init__(
        self,
        index: int,
        g: float,
        h: float,
        parent_key: Optional[Tuple[int, float]],
        duration: float,
        timestamp: float,
    ) -> None:
        """
        Initialize a CAstarNode.

        Args:
            index (int): Node index.
            g (float): Cost from start to this node.
            h (float): Heuristic cost from this node to the goal.
            parent (Optional[int]): Parent node (index, timestamp).
            duration (float): Duration to reach this node from parent.
            timestamp (float): Time at which the node is reached.
        """
        self.index = index
        self.g = g
        self.h = h
        self.parent_key = parent_key
        self.duration = duration
        self.timestamp = timestamp
        self.f = self.g + self.h


    def __lt__(self, other: "CAstarNode") -> bool:
        """Comparison method for priority queue."""
        return self.f < other.f

    def __repr__(self) -> str:
        """String representation of the node."""
        if self.parent_key is None:
            return (
                f"Node {self.index} with None parent, "
                f"duration {self.duration:.2f}, timestamp {self.timestamp:.2f}, "
                f"and f {self.f:.2f}"
            )
        return (
            f"Node {self.index} with parent {self.parent_key[0]}, {self.parent_key[1]:.3f}, "
            f"duration {self.duration:.2f}, timestamp {self.timestamp:.2f}, "
            f"and f {self.f:.2f}"
        )


class CAstar:
    """Cooperative A* algorithm for multi-agent path planning with reservations."""

    def __init__(
        self,
        road_graph: RoadGraph,
        start_states: List[int],
        end_states: List[int],
        vehicles: List[VehicleType],
        average_velocity: float,
        size_buffer: float = 0.0,
        wait_time: float = 0.5,
        time_buffer: float = 0.3,
    ) -> None:
        """
        Initialize the Cooperative A* solver.

        Args:
            road_graph (RoadGraph): The graph of the map.
            start_states (List[int]): List of starting nodes for each agent.
            end_states (List[int]): List of goal nodes for each agent.
            vehicles (List[VehicleType]): List of vehicle objects for each agent.
            average_velocity (float): Average velocity of the vehicles.
            size_buffer (float, optional): Buffer size to inflate vehicle model. Defaults to 0.0.
            wait_time (float, optional): Time to wait at a node. Defaults to 0.5.
            time_buffer (float, optional): Time buffer to avoid close collisions. Defaults to 0.3.
        """
        if not (len(start_states) == len(end_states) == len(vehicles)):
            raise ValueError(
                "The lists of start states, end states, and vehicles must have the same length."
            )

        self.road_graph = road_graph
        self.road_graph_details = nx.get_node_attributes(
            self.road_graph.full_graph, "pos"
        )
        self.start_states = start_states
        self.end_states = end_states
        self.vehicles = vehicles
        self.average_velocity = average_velocity
        self.size_buffer = size_buffer
        self.wait_time = wait_time
        self.time_buffer = time_buffer

        self.debug_closed_set_check = True 
        self.node_visited_list = []
        self.key_visited_list = []

        self.reservation_table: ReservationTable = []

    def get_heuristic(self, state: int, end: int) -> float:
        """
        Compute the heuristic (estimated cost) from the current state to the goal.

        Args:
            state (int): The current node index.
            end (int): The goal node index.

        Returns:
            float: The heuristic cost.
        """
        a = self.road_graph_details[state]
        b = self.road_graph_details[end]
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def round_timestamp(self, timestamp: float) -> float:
        """Round the timestamp to a fixed precision to avoid float precision issues."""
        return round(timestamp, 5)

    def create_swept_polygon(
        self, parent: int, child: int, vehicle: VehicleType
    ) -> Polygon:
        """
        Create a swept polygon between two nodes for collision checking.

        Args:
            parent (int): Index of the parent node.
            child (int): Index of the child node.
            vehicle (VehicleType): The vehicle object.

        Returns:
            Polygon: The swept polygon representing the area occupied during movement.
        """
        parent_pos = self.road_graph_details[parent]
        child_pos = self.road_graph_details[child]
        theta = np.arctan2(child_pos[1] - parent_pos[1], child_pos[0] - parent_pos[0])

        # Construct vehicle at start and end positions
        start_pose = (*parent_pos, theta)
        vehicle.construct_vehicle(start_pose)
        start_polygon = vehicle._vehicle_model

        end_pose = (*child_pos, theta)
        vehicle.construct_vehicle(end_pose)
        end_polygon = vehicle._vehicle_model

        # Create swept area
        swept_area = MultiPolygon([start_polygon, end_polygon]).convex_hull
        swept_buffer = buffer(swept_area, self.size_buffer)

        return swept_buffer

    def is_node_free(
        self,
        parent_node: int,
        child_node: int,
        start_time: float,
        end_time: float,
        vehicle: VehicleType,
    ) -> bool:
        """
        Check if moving from parent_node to child_node is collision-free within the given time window.

        Args:
            parent_node (int): Index of the parent node.
            child_node (int): Index of the child node.
            start_time (float): Time when movement starts.
            end_time (float): Time when movement ends.
            vehicle (VehicleType): The vehicle object.

        Returns:
            bool: True if the movement is collision-free, False otherwise.
        """
        travel_polygon = self.create_swept_polygon(parent_node, child_node, vehicle)
        start_time = start_time - self.time_buffer
        end_time = end_time + self.time_buffer
        for obstacle, start_reserved, end_reserved in self.reservation_table:
            if end_time >= start_reserved and start_time <= end_reserved:
                if travel_polygon.intersects(obstacle):
                    return False
        return True

    def is_not_colliding_with_map(
        self, parent_node: int, child_node: int, vehicle: VehicleType
    ) -> bool:
        """
        Check if the movement between two nodes collides with the map.

        Assumes that the vehicle moves in a straight line between nodes.

        Args:
            parent_node (int): Index of the parent node.
            child_node (int): Index of the child node.
            vehicle (VehicleType): The vehicle object.

        Returns:
            bool: True if the path is collision-free with the map, False otherwise.
        """
        travel_polygon = self.create_swept_polygon(parent_node, child_node, vehicle)
        return travel_polygon.within(self.road_graph.road_map.map)

    def bias_to_prev_path(self, node: int, previous_path: Optional[List[int]]) -> float:
        """
        Calculate a bias factor for a node based on its position in a previous path.

        Args:
            node (int): The node index.
            previous_path (Optional[List[int]]): The previous path for biasing.

        Returns:
            float: A bias factor (>=1), lower if the node is earlier in the previous path.
        """
        bias_mult = 0.5  # Bias scaling factor
        if not previous_path:
            return 1.0  # No scaling
        try:
            index = previous_path.index(node)
            return 1.0 - (bias_mult / (index + 1))
        except ValueError:
            return 1.0  # No scaling for nodes not in the previous path

    def multi_plan(
        self, previous_paths: Optional[List[Optional[List[int]]]] = None
    ) -> List[Optional[List[TimedNode]]]:
        """
        Start the multi-agent planning.

        Args:
            previous_paths (Optional[List[Optional[List[int]]]], optional): List of previous paths for biasing. Defaults to None.

        Returns:
            List[Optional[List[TimedNode]]]: List of paths for each agent.
        """
        paths = []
        if previous_paths is None:
            previous_paths = [None] * len(self.start_states)  # Ensure correct length

        for start, end, vehicle, previous_path in zip(
            self.start_states, self.end_states, self.vehicles, previous_paths
        ):
            path = self.plan(start, end, vehicle, previous_path)
            paths.append(path)

        return paths

    def plan(
        self,
        start: int,
        end: int,
        vehicle: VehicleType,
        previous_path: Optional[List[int]] = None,
    ) -> Optional[List[TimedNode]]:
        """
        Plan a path from start to end for a single agent.

        Args:
            start (int): Index of the starting node.
            end (int): Index of the goal node.
            vehicle (VehicleType): The vehicle object.
            previous_path (Optional[List[int]], optional): Previous path for biasing. Defaults to None.

        Returns:
            Optional[List[TimedNode]]: The planned path as a list of TimedNode objects.
        """
        start_node = self.create_start_node(start, end)
        open_set = []
        heapq.heappush(open_set, start_node)
        came_from: Dict[Tuple[int, float], CAstarNode] = {}
        came_from_key = (start_node.index, self.round_timestamp(start_node.timestamp))
        came_from[came_from_key[0]] = start_node
        closed_set: Set[Tuple[int, float]] = set()

        iteration = 0  # Track iterations for debugging
        max_debug_iterations = 10  # Limit the amount of debug output for tracing
        all_print = np.inf
        
        while open_set:
            if iteration < max_debug_iterations:
                print(f"\n--- Iteration {iteration} ---")
                print(f"open_set size: {len(open_set)}")
                print("open_set contents:", open_set[:5], "...")  # Limit to first few items for readability
            
            current_node = heapq.heappop(open_set)
            current_key = (current_node.index, self.round_timestamp(current_node.timestamp))
            
            if iteration < max_debug_iterations:
                print(f"Popped from open_set: {current_node} with f={current_node.f:.2f}, g={current_node.g:.2f}, h={current_node.h:.2f}")
                print(f"closed_set size: {len(closed_set)}")
            
            if current_key in closed_set:
                if iteration < max_debug_iterations:
                    print("Current node already in closed_set, skipping.")
                continue

            self.node_visited_list.append(current_node)
            self.key_visited_list.append(current_key)

            if len(self.node_visited_list) % 100 == 0:
                print("NODES")
                print([x.index for x in self.node_visited_list])
                print("KEYS")
                print(self.key_visited_list)

            # Check that node has not been added to closed_set already
            if self.debug_closed_set_check:
                assert current_key not in closed_set, f"Node {current_key} is already in closed_set but was re-added from open_set."

            # If already in closed_set, this node has been expanded
            if current_key in closed_set:
                if self.debug_closed_set_check:
                    print(f"Potential cycle detected with node: {current_key}")
                continue
            
            closed_set.add(current_key)
            
            if self.is_goal(current_node, end):
                print("Goal reached!")
                return self.reconstruct_path(came_from, current_node, vehicle, start)
            
            # Expand current node
            if iteration < max_debug_iterations:
                print("Expanding current node:")
            self.expand_node(current_node, open_set, came_from, vehicle, previous_path, end)
            
            iteration += 1

        print(f"Path searching from {start} to {end} nodes failed")
        return None

    def create_start_node(self, start: int, end: int) -> CAstarNode:
        """
        Create the starting node for the search.

        Args:
            start (int): Index of the starting node.
            end (int): Index of the goal node.

        Returns:
            CAstarNode: The starting node.
        """
        h = self.get_heuristic(start, end)
        return CAstarNode(
            index=start, g=0.0, h=h, parent_key=None, duration=0.0, timestamp=0.0
        )

    def select_current_node(self, open_set: List[CAstarNode]) -> CAstarNode:
        """
        Select the node with the lowest f value from the open set.

        Args:
            open_set (List[CAstarNode]): The open set.

        Returns:
            CAstarNode: The node with the lowest f value.
        """
        return heapq.heappop(open_set)

    def is_goal(self, node: CAstarNode, end: int) -> bool:
        """
        Check if the current node is the goal.

        Args:
            node (CAstarNode): The current node.
            end (int): Index of the goal node.

        Returns:
            bool: True if the current node is the goal, False otherwise.
        """
        return node.index == end

    def expand_node(
        self,
        current_node: CAstarNode,
        open_set: List[CAstarNode],
        came_from: Dict[Tuple[int, float], CAstarNode],
        vehicle: VehicleType,
        previous_path: Optional[List[int]],
        end: int,
    ) -> None:
        """
        Expand the current node by evaluating its neighbors.

        Args:
            current_node (CAstarNode): The node being expanded.
            open_set (List[CAstarNode]): The open set.
            came_from (Dict[Tuple[int, float], CAstarNode]): Dictionary mapping nodes to their parents.
            vehicle (VehicleType): The vehicle object.
            previous_path (Optional[List[int]]): Previous path for biasing.
            end: End node index.
        """
        neighbor_indices = self.road_graph.full_graph.neighbors(current_node.index)
        for neighbor_index in neighbor_indices:
            self.evaluate_neighbor(
                current_node,
                neighbor_index,
                open_set,
                came_from,
                vehicle,
                previous_path,
                end,
            )

        # Add wait action
        wait_node_key = (
            current_node.index,
            self.round_timestamp(current_node.timestamp + self.wait_time),
        )
        if wait_node_key not in came_from:
            wait_node = CAstarNode(
                index=current_node.index,
                g=current_node.g + self.wait_time * self.average_velocity,
                h=current_node.h,
                parent_key=(
                    current_node.index,
                    self.round_timestamp(current_node.timestamp),
                ),
                duration=self.wait_time,
                timestamp=current_node.timestamp + self.wait_time,
            )
            # heapq.heappush(open_set, wait_node)
            # came_from[wait_node_key] = wait_node
            # print(f"Added wait_node to open_set: {wait_node}")

    def evaluate_neighbor(
        self,
        current_node: CAstarNode,
        neighbor_index: int,
        open_set: List[CAstarNode],
        came_from: Dict[Tuple[int, float], CAstarNode],
        vehicle: VehicleType,
        previous_path: Optional[List[int]],
        end: int,
    ) -> None:
        """
        Evaluate a neighbor node and update the open set and came_from dict.

        Args:
            current_node (CAstarNode): The current node.
            neighbor_index (int): The index of the neighbor node.
            open_set (List[CAstarNode]): The open set.
            came_from (Dict[Tuple[int, float], CAstarNode]]): Dictionary mapping nodes to their parents.
            vehicle (VehicleType): The vehicle object.
            previous_path (Optional[List[int]]): Previous path for biasing.
            end: End node index.
        """
        edge_data = self.road_graph.full_graph.get_edge_data(
            current_node.index, neighbor_index
        )
        if edge_data is None:
            return  # No edge between current node and neighbor

        edge_weight = edge_data["weight"] * self.bias_to_prev_path(
            neighbor_index, previous_path
        )
        travel_duration = edge_weight / self.average_velocity
        travel_start_time = current_node.timestamp
        travel_end_time = travel_start_time + travel_duration

        neighbor_node_key = (neighbor_index, self.round_timestamp(travel_end_time))
        tentative_g = current_node.g + edge_weight

        # Cost comparison and pruning
        existing_node = came_from.get(neighbor_node_key[0])
        if existing_node and tentative_g >= existing_node.g:
            return  # Existing path to neighbor is cheaper or equal; skip this path

        # Check for collisions
        if not self.is_node_free(
            current_node.index,
            neighbor_index,
            travel_start_time,
            travel_end_time,
            vehicle,
        ):
            return

        h = self.get_heuristic(neighbor_index, end)
        neighbor_node = CAstarNode(
            index=neighbor_index,
            g=tentative_g,
            h=h,
            parent_key=(current_node.index, self.round_timestamp(current_node.timestamp)),
            duration=travel_duration,
            timestamp=travel_end_time,
        )

        # Add or update the node in open_set and came_from
        heapq.heappush(open_set, neighbor_node)
        came_from[neighbor_node_key[0]] = neighbor_node

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, float], CAstarNode],
        current_node: CAstarNode,
        vehicle: VehicleType,
        start: int,
    ) -> List[TimedNode]:
        """
        Reconstruct the path from start to goal.

        Args:
            came_from (Dict[Tuple[int, float], CAstarNode]): Dictionary mapping nodes to their parents.
            current_node (CAstarNode): The goal node.
            vehicle (VehicleType): The vehicle object.
            start (int): Index of the starting node.

        Returns:
            List[TimedNode]: The reconstructed path.
        """
        path = []
        while current_node.parent_key is not None:
            parent_node = came_from[current_node.parent_key[0]]
            timed_node = TimedNode(
                index=current_node.index,
                enter_time=parent_node.timestamp,
                exit_time=current_node.timestamp,
            )
            path.append(timed_node)

            # Update reservation table
            self.add_to_reservation_table(
                parent_index=parent_node.index,
                child_index=current_node.index,
                timed_node=timed_node,
                vehicle=vehicle,
            )

            current_node = parent_node

        # Add the start node
        path.append(TimedNode(index=start, enter_time=-1.0, exit_time=0.0))
        path.reverse()
        return path

    def add_to_reservation_table(
        self,
        parent_index: int,
        child_index: int,
        timed_node: TimedNode,
        vehicle: VehicleType,
    ) -> None:
        """
        Add a movement to the reservation table to prevent future collisions.

        Args:
            parent_index (int): Index of the parent node.
            child_index (int): Index of the child node.
            timed_node (TimedNode): Node with timing information.
            vehicle (VehicleType): The vehicle object.
        """
        if parent_index == child_index:
            # Waiting at the node
            node_pos = self.road_graph_details[child_index]
            vehicle.construct_vehicle((*node_pos, 0.0))  # Orientation doesn't matter
            occupied_area = buffer(vehicle._vehicle_model, self.size_buffer)
        else:
            # Movement between nodes
            occupied_area = self.create_swept_polygon(
                parent_index, child_index, vehicle
            )
        self.reservation_table.append(
            (occupied_area, timed_node.enter_time, timed_node.exit_time)
        )
