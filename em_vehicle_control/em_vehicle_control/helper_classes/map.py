import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import CirclePolygon as MplCirclePolygon, FancyArrowPatch
import matplotlib.patches as patches
from typing import Tuple, Optional, List, Dict
import numpy as np
import itertools
import networkx as nx
from scipy.spatial import KDTree
from scipy.interpolate import make_interp_spline, CubicSpline
from copy import deepcopy
from enum import Enum


class RoadSegment:
    """A road segment describes the full rectangular part of the road"""

    # Note: This class only works for axis_aligned roads

    def __init__(
        self,
        point_A: Tuple[float, float],
        point_B: Tuple[float, float],
        invert_direction: bool = False,
    ):
        """
        pointA: (x, y) coordinate of bottom left point
        pointB: (x, y) coordinate of top right point
        invert_direction: False is the long side of the road is the direction the car moves in, true if otherwise
        """
        cornerBL = (point_A[0], point_A[1])
        cornerBR = (point_B[0], point_A[1])
        cornerTL = (point_A[0], point_B[1])
        cornerTR = (point_B[0], point_B[1])
        self.x_min = point_A[0]
        self.x_max = point_B[0]
        self.y_min = point_A[1]
        self.y_max = point_B[1]
        self.x_length = np.abs(point_B[0] - point_A[0])
        self.y_length = np.abs(point_B[1] - point_A[1])
        self.road_segment_polygon = Polygon((cornerBL, cornerBR, cornerTR, cornerTL))
        if not self.road_segment_polygon.is_valid:
            print(
                f"Error: The outer boundary given is not valid for road {point_A}, {point_B}"
            )

        if self.x_length > self.y_length:
            # if the length along the x-axis is longer than the length along the y-axis
            # i.e., the road is going from left to right
            self.direction = -1
        else:
            self.direction = 1  # the road is going up to down

        if invert_direction:
            self.direction *= -1


class RoadMap:
    """
    The road map is the full map consisting of parallel and perpendicular road segments

    """

    def __init__(self, roads: list[RoadSegment], debug: bool = False) -> None:
        self.roads = roads
        self.debug = debug
        road_intersections = self.get_intersections()
        self.road_intersections = road_intersections
        self.road_intersections_poly = [x[0] for x in road_intersections]
        self.stations = []  # TODO should this be a dictionary with names?
        self.map = shapely.unary_union([road.road_segment_polygon for road in roads])

    def get_intersections(self) -> list[Tuple[Polygon, RoadSegment, RoadSegment]]:
        """
        Function that iteratively looks through all roads and identify intersection points
        Returns a list of tuples that include the Polygon of the intersection
        and the indices of the 2 roads that creates the intersection
        """
        road_intersections = []
        for i in range(len(self.roads)):
            for j in range(i + 1, len(self.roads)):
                if shapely.intersects(
                    self.roads[i].road_segment_polygon,
                    self.roads[j].road_segment_polygon,
                ):
                    road_intersection = shapely.intersection(
                        self.roads[i].road_segment_polygon,
                        self.roads[j].road_segment_polygon,
                    )
                    if self.debug:
                        print(
                            f"Intersection ({road_intersection}) created by roads ({i}, {self.roads[i]}) and ({j}, {self.roads[j]})"
                        )
                    road_intersections.append((road_intersection, i, j))
        return road_intersections

    def add_station(
        self,
        location: Tuple[float, float],
        orientation: float,
        radius: Optional[float] = 0.05,
    ) -> Tuple[Polygon, float]:
        """
        Method to add stations, signfying the end goal of the robot
        location: (x, y) coordinates of the station
        orientation: final orientation of the vehicle
        radius: radius describing the tolerance of the end point (default at 5cm radius)
        """
        station = Point(location)
        station.buffer(radius, resolution=100)
        self.stations.append((station, location, orientation, radius))

    def get_local_map(self, locations: List[Tuple[float, float]], radius: float = 0.5) -> Polygon:
        """
        points: (x,y) list of points of interest
        radius: radius of important region around each point
        return: intersection between convex hull of all important regions and the map
        """
        buffers = [Point(loc).buffer(radius) for loc in locations]
        union_buffers = unary_union(buffers)
        region_of_interest = union_buffers.convex_hull
        local_map = self.map.intersection(region_of_interest)
        return local_map

    @staticmethod
    def shapely_to_mplpolygons(
        shapely_polygon: Polygon, colour: str = "lightblue"
    ) -> Tuple[MplPolygon, MplPolygon]:
        x, y = shapely_polygon.exterior.xy
        exterior_polygon = MplPolygon(
            list(zip(x, y)), closed=True, edgecolor="black", facecolor=colour, alpha=0.5
        )
        hole_polygons = []
        for hole in shapely_polygon.interiors:
            x, y = hole.xy
            hole_polygons.append(
                MplPolygon(
                    list(zip(x, y)),
                    closed=True,
                    edgecolor="black",
                    facecolor="white",
                    alpha=1,
                )
            )
        return exterior_polygon, hole_polygons

    def visualise(
        self,
        show_intersections: bool = False,
        show_stations: bool = True,
        graph: Optional[nx.Graph] = None,
        path: Optional[List] = None,
    ) -> None:
        """Visualises the built map"""
        _, ax = plt.subplots()
        exterior_polygon, interior_polygons = self.shapely_to_mplpolygons(self.map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in interior_polygons:
            ax.add_patch(hole_polygon)
        if show_intersections:
            for road_intersection in self.road_intersections:
                rd_int_polygon, _ = self.shapely_to_mplpolygons(
                    road_intersection[0], "purple"
                )
                ax.add_patch(rd_int_polygon)
        if show_stations:
            for station in self.stations:
                station_polygon = MplCirclePolygon(
                    station[1],
                    radius=station[3],
                    resolution=100,
                    facecolor="green",
                    alpha=0.5,
                )
                ax.add_patch(station_polygon)
                # print(station[1][0], station[1][1], station[1][0]+station[3]*np.cos(station[2]), station[1][1]+station[3]*np.sin(station[2]))
                ax.arrow(
                    station[1][0],
                    station[1][1],
                    station[3] * np.cos(station[2]),
                    station[3] * np.sin(station[2]),
                    width=station[3] / 10,
                )
        if graph is not None:
            pos = nx.get_node_attributes(graph, "pos")
            lane_labels = nx.get_node_attributes(graph, "lane")

            # Group nodes by lane
            north_nodes = [node for node, lane in lane_labels.items() if lane == "north"]
            south_nodes = [node for node, lane in lane_labels.items() if lane == "south"]
            east_nodes = [node for node, lane in lane_labels.items() if lane == "east"]
            west_nodes = [node for node, lane in lane_labels.items() if lane == "west"]

            # Draw nodes with colors based on lane labels
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=north_nodes,
                node_size=25,
                node_color="blue",
                label="Right Lane",
                ax=ax
            )
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=south_nodes,
                node_size=25,
                node_color="red",
                label="Left Lane",
                ax=ax
            )
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=east_nodes,
                node_size=25,
                node_color="green",
                label="Right Lane",
                ax=ax
            )
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=west_nodes,
                node_size=25,
                node_color="violet",
                label="Left Lane",
                ax=ax
            )

            for u, v, data in graph.edges(data=True):
                if "geometry" in data:
                    line = data["geometry"]  # This should be a LineString
                    x, y = line.xy  # Get the x, y coordinates from the LineString
                    ax.plot(x, y, color="gray")
                    # # Get the midpoint of the edge
                    # midpoint_x = (x[0] + x[-1]) / 2
                    # midpoint_y = (y[0] + y[-1]) / 2

                    # # Get the edge weight
                    # weight = data.get("weight", None)
                    # if weight is not None:
                    #     ax.text(midpoint_x, midpoint_y, f"{weight:.2f}", fontsize=8, color="black")
            if path is not None:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(
                    graph, pos, edgelist=path_edges, edge_color="orange"
                )
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=path,
                    node_size=5,
                    node_color="orange",
                )
                
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="black")

        ax.axis("scaled")
        plt.show()


class RoadGraph:
    def __init__(
        self,
        min_length: float,
        min_width: float,
        road_map: RoadMap,
        visualise: bool = False,
    ) -> None:
        """
        Class that creates grid of dots, conencts the edges between nearest vertices
        """
        self.min_length = min_length
        self.min_width = min_width
        self.road_map = road_map
        self.visualise = visualise
        self.buffer_distance = 0.1

        self.max_length = 2 * self.min_length
        self.vertices = []
        self.global_kdtree = None
        self.global_graph: nx.Graph = None
        self.full_graph: nx.Graph = None

        self.make_vertices()

    def _create_graph(
        self, points: list, min_length: float, max_length: float
    ) -> nx.Graph:
        """
        points: list of (x,y) points
        min_length: minimum edge length (L2) between 2 points
        max_length: maximum edge length between 2 points
        returns the graph of edges and vertices
        """
        self.global_kdtree = KDTree(points)
        self.global_graph = nx.Graph()
        for idx, point in enumerate(points):
            self.global_graph.add_node(idx, pos=point)

        # for every point, search for their neighbour
        for idx, point in enumerate(points):
            indices = self.global_kdtree.query_ball_point(point, max_length)
            for neighbour_idx in indices:
                if neighbour_idx == idx:
                    # avoid calling itself
                    continue
                dist = np.linalg.norm(np.array(point) - np.array(points[neighbour_idx]))
                if not min_length <= dist <= max_length:
                    # outside allowable lengths
                    continue
                candidate_edge = LineString([points[neighbour_idx], point])
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue
                # TODO: decide on weights
                self.global_graph.add_edge(idx, neighbour_idx, weight=dist)
        return self.global_graph

    def make_vertices(self) -> None:
        """
        min_length: the minimum safe distance between vertices parallel to the direction of the road
        min_width: the minimum safe distance between vertices perpendicular to the direction of the road
        visualise: plots the grid on the map
        min_length should be longer than min_width
        """
        buffered_map = self.road_map.map.buffer(-self.buffer_distance)
        for road in self.road_map.roads:
            if road.direction == -1:  # x longer than y
                # x is the length and y is the width
                min_x_dist = self.min_length
                min_y_dist = self.min_width
            else:
                min_x_dist = self.min_width
                min_y_dist = self.min_length
            x_segments = np.floor(road.x_length / min_x_dist).astype(int)
            y_segments = np.floor(road.y_length / min_y_dist).astype(int)
            x_seg_points = [
                (road.x_max - road.x_min) / (x_segments * 2) * (2 * i + 1) + road.x_min
                for i in range(x_segments)
            ]
            y_seg_points = [
                (road.y_max - road.y_min) / (y_segments * 2) * (2 * i + 1) + road.y_min
                for i in range(y_segments)
            ]
            grid_points = list(itertools.product(x_seg_points, y_seg_points))
            for point in grid_points:
                if Point(point).within(buffered_map):
                    self.vertices.append(point)

        self.global_graph = self._create_graph(
            self.vertices, self.min_length, self.max_length
        )

        if self.visualise:
            self.road_map.visualise(True, True, self.global_graph)

    def get_N_nearest_vertices(
        self, point: Tuple, N: int = 1
    ) -> list[int]:
        """
        point: Queried point (x,y)
        N: number of points to return
        returns N nearest vertices to the queried point
        """
        _, idx = self.global_kdtree.query(point, N)
        return idx

    def add_edges_between_consecutive_rings(
        self,
        local_graph: nx.Graph,
        current_side: List[Tuple[float, float]],
        next_side: List[Tuple[float, float]],
        point_indices: dict[int : Tuple[float, float]],
        local_graph_bias: float,
    ) -> nx.Graph:
        """
        Add edges between points on the current side and the next side of the local graph.

        local_graph: The networkx graph representing the local graph.
        current_side: List of points on the current ring's side (top, right, bottom, or left).
        next_side: List of points on the next ring's side.
        point_indices: Dictionary mapping points to their indices in the graph.
        local_graph_bias: Bias factor for the local graph's edge weights.
        """
        for i in range(len(current_side)):  # Avoid going out of bounds
            candidate_edge = LineString([next_side[i], current_side[i]])
            if not candidate_edge.within(self.road_map.map):
                continue
            # Connect current point to two closest points in the next ring
            local_graph.add_edge(
                point_indices[current_side[i]],
                point_indices[next_side[i]],
                weight=local_graph_bias
                * np.linalg.norm(np.array(next_side[i]) - np.array(current_side[i])),
            )
        for i in range(len(current_side)):
            candidate_edge = LineString([next_side[i + 1], current_side[i]])
            if not candidate_edge.within(self.road_map.map):
                continue
            local_graph.add_edge(
                point_indices[current_side[i]],
                point_indices[next_side[i + 1]],
                weight=local_graph_bias
                * np.linalg.norm(
                    np.array(next_side[i + 1]) - np.array(current_side[i])
                ),
            )

        return local_graph

    def add_edges_between_following_rings(
        self,
        local_graph: nx.Graph,
        current_side: List[Tuple[float, float]],
        following_side: List[Tuple[float, float]],
        point_indices: dict[int : Tuple[float, float]],
        local_graph_bias: float,
    ) -> nx.Graph:
        """
        Add edges between points on the current side and the following side of the local graph.

        local_graph: The networkx graph representing the local graph.
        current_side: List of points on the current ring's side (top, right, bottom, or left).
        following_side: List of points on the following ring's side.
        point_indices: Dictionary mapping points to their indices in the graph.
        local_graph_bias: Bias factor for the local graph's edge weights.
        """
        for i in range(len(current_side)):  # Avoid going out of bounds
            candidate_edge = LineString([following_side[i], current_side[i]])
            if not candidate_edge.within(self.road_map.map):
                continue
            # Connect current point to two closest points in the next ring
            local_graph.add_edge(
                point_indices[current_side[i]],
                point_indices[following_side[i + 1]],
                weight=local_graph_bias
                * np.linalg.norm(
                    np.array(following_side[i + 1]) - np.array(current_side[i])
                ),
            )

        return local_graph

    def generate_local_graph(
        self,
        pose: Tuple[float, float, float],
        step_size: float,
        num_rings: int = 5,
        multiplier: float = 1.3,
    ) -> nx.Graph:
        """
        pose: tuple containing the x, y, yaw values of the robot
        step_size: step size of each ring in metres
        num_rings: number of rings of points surrounding the robot
        multiplier: scales the rings distance by multiplier ^ ring number

        returns: local graph generated around
        """
        # first generate all points about the origin
        # then translate and rotate all points
        local_graph = nx.Graph()
        current_index = 0
        origin = (pose[0], pose[1])
        local_graph.add_node(current_index, pos=origin)
        c_theta = np.cos(pose[2])
        s_theta = np.sin(pose[2])
        rings_of_points = []
        point_indices = {}
        current_index += 1
        local_graph_bias = 0.5
        for i in range(1, num_rings + 1):
            sides = [[], [], [], []]  # top, right, bottom, left
            side_transforms = [
                (
                    lambda i, j: (
                        (-i + 2 * j) * step_size * multiplier**i,
                        i * step_size * multiplier**i * 0.6,
                    )
                ),  # Top
                (
                    lambda i, j: (
                        i * step_size * multiplier**i,
                        (i - 2 * j) * step_size * multiplier**i * 0.6,
                    )
                ),  # Right
                (
                    lambda i, j: (
                        (i - 2 * j) * step_size * multiplier**i,
                        -i * step_size * multiplier**i * 0.6,
                    )
                ),  # Bottom
                (
                    lambda i, j: (
                        -i * step_size * multiplier**i,
                        (-i + 2 * j) * step_size * multiplier**i * 0.6,
                    )
                ),  # Left
            ]
            for j in range(i):
                for side_idx, transform in enumerate(side_transforms):
                    x_temp, y_temp = transform(i, j)

                    transformed_point = (
                        x_temp * c_theta - y_temp * s_theta + pose[0],
                        x_temp * s_theta + y_temp * c_theta + pose[1],
                    )
                    sides[side_idx].append(transformed_point)
            rings_of_points.append(sides)

            for side_points in sides:
                for point in side_points:
                    point_indices[point] = current_index
                    local_graph.add_node(
                        current_index, pos=point
                    )  # Add node to the graph with index
                    current_index += 1
        for points in rings_of_points[0]:
            # first add connections to origin
            local_graph.add_edge(
                0,
                point_indices[points[0]],
                weight=local_graph_bias * np.linalg.norm(np.array(points[0])),
            )
        if num_rings >= 2:
            for points in rings_of_points[1]:
                local_graph.add_edge(
                    0,
                    point_indices[points[1]],
                    weight=local_graph_bias * np.linalg.norm(np.array(points[1])),
                )

        sides = [x for x in range(4)]
        for ring_idx in range(len(rings_of_points) - 1):
            # Unpack the current ring and the next ring
            current_ring = rings_of_points[ring_idx]
            next_ring = rings_of_points[ring_idx + 1]
            following_ring = None
            if ring_idx < len(rings_of_points) - 2:
                following_ring = rings_of_points[ring_idx + 2]

            # Loop over each side (top, right, bottom, left)
            for side_idx in sides:
                current_side = current_ring[side_idx]
                next_side = next_ring[side_idx]

                local_graph = self.add_edges_between_consecutive_rings(
                    local_graph,
                    current_side,
                    next_side,
                    point_indices,
                    local_graph_bias,
                )
                if following_ring is not None:
                    following_side = following_ring[side_idx]
                    local_graph = self.add_edges_between_following_rings(
                        local_graph,
                        current_side,
                        following_side,
                        point_indices,
                        local_graph_bias,
                    )

            # Handle corner connections
            for i in range(4):
                candidate_edge = LineString(
                    [current_ring[i][0], next_ring[(i - 1) % 4][-1]]
                )
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue
                local_graph.add_edge(
                    point_indices[
                        current_ring[i][0]
                    ],  # First point in each side of the current ring
                    point_indices[
                        next_ring[(i - 1) % 4][-1]
                    ],  # Last point in the first anticlockwise side of the next ring
                    weight=local_graph_bias
                    * np.linalg.norm(
                        np.array(current_ring[i][0])
                        - np.array(next_ring[(i - 1) % 4][-1])
                    ),
                )
            for i in range(4):
                if following_ring is not None:
                    candidate_edge = LineString(
                        [current_ring[i][0], following_ring[(i - 1) % 4][-1]]
                    )
                    if not candidate_edge.within(self.road_map.map):
                        # motion path is disallowed
                        continue
                    local_graph.add_edge(
                        point_indices[
                            current_ring[i][0]
                        ],  # First point in each side of the current ring
                        point_indices[
                            following_ring[(i - 1) % 4][-1]
                        ],  # Last point in the first anticlockwise side of the next ring
                        weight=local_graph_bias
                        * np.linalg.norm(
                            np.array(current_ring[i][0])
                            - np.array(following_ring[(i - 1) % 4][-1])
                        ),
                    )

        # pos = nx.get_node_attributes(local_graph, 'pos')
        # labels = {node: f"({round(x, 2)}, {round(y, 2)})" for node, (x, y) in pos.items()}
        # nx.draw(local_graph, pos, labels=labels, with_labels=False, node_size=10)
        # ax = plt.gca()  # Get current axes
        # for edge in local_graph.edges():
        #     # Get positions of the nodes at the ends of the edge
        #     start_pos = pos[edge[0]]
        #     end_pos = pos[edge[1]]

        #     # Create a FancyArrowPatch for the edge
        #     arrow = FancyArrowPatch(start_pos, end_pos, arrowstyle='->', color='grey', mutation_scale=30.0)
        #     # ax.add_patch(arrow)
        # plt.show()
        return local_graph

    def combine_graph(
        self,
        main_graph: nx.graph,
        local_graph: nx.graph,
        min_length: float,
        max_length: float,
    ) -> nx.graph:

        index_shifter = max(main_graph.nodes) + 1
        for local_idx, local_point in local_graph.nodes(data="pos"):
            main_graph.add_node(local_idx + index_shifter, pos=local_point)
        for edge in local_graph.edges(data=True):
            local_idx_1, local_idx_2, edge_data = edge
            main_graph.add_edge(
                local_idx_1 + index_shifter, local_idx_2 + index_shifter, **edge_data
            )
        points = {node: data["pos"] for node, data in main_graph.nodes(data=True)}
        for local_idx, local_point in local_graph.nodes(data="pos"):
            indices = self.global_kdtree.query_ball_point(local_point, max_length)
            # print(f"For {local_point}, found neighbours: {indices}")
            for neighbour_idx in indices:
                # print(f"looking at neighbour {points[neighbour_idx]}")
                dist = np.linalg.norm(
                    np.array(local_point) - np.array(points[neighbour_idx])
                )
                if not min_length <= dist <= max_length:
                    # print("Distance outside annular")
                    # outside allowable lengths
                    continue
                candidate_edge = LineString([points[neighbour_idx], local_point])
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue
                # TODO: decide on weights
                main_graph.add_edge(
                    local_idx + index_shifter, neighbour_idx, weight=dist
                )
        return main_graph, index_shifter

    def loop_graph(
        self,
        poses,
        step_size: float,
        num_rings: int = 4,
        multiplier: float = 1.3,
    ):
        self.full_graph = deepcopy(self.global_graph)
        self.pose_indices = []
        for pose in poses:
            local_graph = self.generate_local_graph(
                pose, step_size, num_rings, multiplier
            )
            self.full_graph, pose_idx = self.combine_graph(
                self.full_graph, local_graph, self.min_length, self.max_length
            )
            self.pose_indices.append(pose_idx)

        # self.road_map.visualise(True, True, self.full_graph)


class RoadDirection(Enum):
    LONG_X = -1
    LONG_Y = 1

class RoadTrack:
    """
    A class to generate grid points for a road network based on the assumption
    that each road has two tracks. It handles the creation of vertices and edges
    for roads and intersections, and represents the network as a graph.
    """

    def __init__(self, road_map: RoadMap):
        """
        Initializes the RoadTrack object with a given road map.

        Parameters:
            road_map (RoadMap): An object containing the map, roads, and intersections.
        """
        self.road_map = road_map
        self.map = road_map.map
        self.roads = road_map.roads
        self.road_intersections = road_map.road_intersections_poly
        self.road_intersections_data = road_map.road_intersections
        self.vertices: List[Tuple[Point, str]] = []
        self.vertex_to_index: Dict[Point, int] = {}
        self.edges: Dict[Tuple[Point, Point], LineString] = {}
        self.min_seg_length = 0.4
        self.reverse_lane_penalty_mult = 10

        self.generate_grid_points()
        self.full_graph = self.generate_nx_graph()
        self.global_kdtree = KDTree([vertex[0].coords[0] for vertex in self.vertices])

        # Dynamic graph for setting static obstacles
        self.dynamic_graph = self.full_graph.copy()

    def generate_grid_points(self):
        """
        Generates grid points for the road network by processing each road and intersection.
        """
        for road in self.roads:
            self._generate_road_points(road)
        road_pts_kd_tree = KDTree([vertex[0].coords[0] for vertex in self.vertices])
        for road_intersection, road1, road2 in self.road_intersections_data:
            self._generate_intersection_points(
                road_intersection,
                self.roads[road1],
                self.roads[road2],
                road_pts_kd_tree,
            )

    def generate_nx_graph(self) -> nx.DiGraph:
        """
        Generates a directed graph representing the road network.

        Returns:
            nx.DiGraph: The generated directed graph.
        """
        graph = nx.DiGraph()
        for idx, (vertex, lane) in enumerate(self.vertices):
            graph.add_node(idx, pos=(vertex.x, vertex.y), lane=lane)

        for (pt_A, pt_B), line in self.edges.items():
            self.add_weighted_edge(graph, pt_A, pt_B, line, self.vertex_to_index, self.vertices, self.reverse_lane_penalty_mult)
            self.add_weighted_edge(graph, pt_B, pt_A, line, self.vertex_to_index, self.vertices, self.reverse_lane_penalty_mult)

        return graph
    
    def block_nodes_within_obstacle(self, obstacle: Polygon) -> List[int]:
        """
        Blocks all nodes within a given polygon

        Args:
            obstacle (Polygon): Region defined as inaccessible

        Returns:
            (List[int]): list of blocked nodes indices
        """
        min_x, min_y, max_x, max_y = obstacle.bounds

        candidate_indices = self.global_kdtree.query_ball_point(
            [((min_x + max_x) / 2, (min_y + max_y) / 2)], 
            r=max(max_x - min_x, max_y - min_y)
        )

        blocked_nodes = []
        # TODO: Block all nodes in AABB or block nodes only within polygon?
        for idx in candidate_indices[0]:
            node_pos = self.dynamic_graph.nodes[idx]["pos"]
            if obstacle.contains(Point(node_pos)):
                self._block_node(idx)
                blocked_nodes.append(idx)
        
        return blocked_nodes
        
    def _block_node(self, node_idx: int) -> None:
        """
        Blocks a node by marking all connected edges as inaccessible

        Args:
            node_idx (int): node index to block
        """
        for neighbor in self.dynamic_graph.successors(node_idx):
            self.dynamic_graph[node_idx][neighbor]["weight"] = float("inf")
        for neighbor in self.dynamic_graph.predecessors(node_idx):
            self.dynamic_graph[neighbor][node_idx]["weight"] = float("inf")
    
    def reset_dynamic_graph(self):
        """
        Resets dynamic graphs of all blocked weights
        """
        self.dynamic_graph = self.full_graph.copy()
    
    def add_weighted_edge(self, graph, pt_A, pt_B, line, vertex_to_index, vertices, reverse_lane_penalty_mult):
        idx_A = vertex_to_index[pt_A]
        idx_B = vertex_to_index[pt_B]
        lane_B = vertices[idx_B][1]
        weight = line.length

        # Apply penalty if moving against the lane direction
        if lane_B == "north" and pt_B.y < pt_A.y:
            weight *= reverse_lane_penalty_mult 
        elif lane_B == "south" and pt_B.y > pt_A.y:
            weight *= reverse_lane_penalty_mult 
        elif lane_B == "east" and pt_B.x < pt_A.x:
            weight *= reverse_lane_penalty_mult 
        elif lane_B == "west" and pt_B.x > pt_A.x:
            weight *= reverse_lane_penalty_mult 

        graph.add_edge(
            idx_A,
            idx_B,
            weight=weight,
            geometry=line,
        )

    def get_N_nearest_vertices(self, point: Tuple[float, float], N: int = 1) -> List[int]:
        """
        Retrieves the indices of the N nearest vertices to a given point.

        Parameters:
            point (Tuple[float, float]): The queried point (x, y).
            N (int): The number of nearest vertices to return.

        Returns:
            List[int]: Indices of the N nearest vertices.
        """
        _, idx = self.global_kdtree.query(point, N)
        return idx

    def _get_bezier_control_points(self, start: Point, end: Point, direction: int) -> Tuple[Point, Point]:
        """
        Calculates control points for a cubic Bezier curve between two points.

        Parameters:
            start (Point): The starting point of the curve.
            end (Point): The ending point of the curve.
            direction (int): The direction of the curve.
                - 1: Vertical line (y-axis aligned)
                - -1: Horizontal line (x-axis aligned)
                - 0: Diagonal or curved connection, inflecting towards the center

        Returns:
            Tuple[Point, Point]: The two control points for the Bezier curve.
        """
        if direction == -1:
            cp1 = Point((end.x - start.x) / 2 + start.x, start.y)
            cp2 = Point((end.x - start.x) / 2 + start.x, end.y)
        elif direction == 1:
            cp1 = Point(start.x, (end.y - start.y) / 2 + start.y)
            cp2 = Point(end.x, (end.y - start.y) / 2 + start.y)
        elif direction == 0:
            corner = Point(start.x, end.y)
            cp1 = Point((start.x + corner.x) / 2, start.y)
            cp2 = Point(corner.x, (corner.y + end.y) / 2)
        else:
            raise ValueError("Invalid direction for Bezier control points")
        return cp1, cp2

    def bezier_curve(self, start: Point, end: Point, direction: int, num_points: int = 30) -> LineString:
        """
        Generates a cubic Bezier curve between two points.

        Parameters:
            start (Point): The starting point of the curve.
            end (Point): The ending point of the curve.
            direction (int): The direction indicator for curve shaping
            num_points (int): The number of points to sample along the curve.

        Returns:
            LineString: The LineString representing the Bezier curve.
        """
        cp1, cp2 = self._get_bezier_control_points(start, end, direction)
        t_values = np.linspace(0, 1, num_points)
        x_start, y_start = start.x, start.y
        x_cp1, y_cp1 = cp1.x, cp1.y
        x_cp2, y_cp2 = cp2.x, cp2.y
        x_end, y_end = end.x, end.y

        curve_x = (
            (1 - t_values) ** 3 * x_start
            + 3 * (1 - t_values) ** 2 * t_values * x_cp1
            + 3 * (1 - t_values) * t_values ** 2 * x_cp2
            + t_values ** 3 * x_end
        )
        curve_y = (
            (1 - t_values) ** 3 * y_start
            + 3 * (1 - t_values) ** 2 * t_values * y_cp1
            + 3 * (1 - t_values) * t_values ** 2 * y_cp2
            + t_values ** 3 * y_end
        )
        curve_points = list(zip(curve_x, curve_y))
        return LineString(curve_points)

    def _add_vertex(self, pt: Point, label: str):
        """
        Adds a vertex with a lane label to the vertices list if it's not already present.

        Parameters:
            pt (Point): The point to add as a vertex.
            label (str): The lane label for the point, either "right" or "left".
        """
        if pt not in self.vertex_to_index:
            self.vertex_to_index[pt] = len(self.vertices)
            self.vertices.append((pt, label))

    def _add_edge(self, start: Point, end: Point, geometry: LineString):
        """
        Adds an edge to the edges dictionary.

        Parameters:
            start (Point): The starting point of the edge.
            end (Point): The ending point of the edge.
            geometry (LineString): The geometry of the edge.
        """
        self.edges[(start, end)] = geometry

    def _add_points_road(self, pt_A: Point, pt_B: Point, pt_C: Point, pt_D: Point, direction: int):
        """
        Adds points and edges for a road segment.

        Parameters:
            pt_A (Point): Starting point on track 1.
            pt_B (Point): Starting point on track 2.
            pt_C (Point): Ending point on track 1.
            pt_D (Point): Ending point on track 2.
            direction (int): Direction indicator for Bezier curves.
        """
        # for pt in [pt_A, pt_B, pt_C, pt_D]:
        #     self._add_vertex(pt)

        self._add_edge(pt_A, pt_C, LineString([pt_A, pt_C]))
        self._add_edge(pt_B, pt_D, LineString([pt_B, pt_D]))
        self._add_edge(pt_A, pt_D, self.bezier_curve(pt_A, pt_D, direction))
        self._add_edge(pt_B, pt_C, self.bezier_curve(pt_B, pt_C, direction))

    def _add_points_road_intersection(
        self, pt_1: Point, pt_2: Point, road_pts_kdtree: KDTree, direction: str
    ):
        """
        Adds points and edges for an intersection segment.

        Parameters:
            pt_1 (Point): First point on the intersection boundary.
            pt_2 (Point): Second point on the intersection boundary.
            road_pts_kdtree (KDTree): KDTree for nearest neighbor search.
            direction (str): Direction to search for neighboring points.
        """
        pt_a = self._get_nearest_direction_neighbor(pt_1, road_pts_kdtree, direction)
        pt_b = self._get_nearest_direction_neighbor(pt_2, road_pts_kdtree, direction)
        self._add_edge(pt_1, pt_a, LineString([pt_1, pt_a]))
        self._add_edge(pt_2, pt_b, LineString([pt_2, pt_b]))

    def _generate_road_points(self, road):
        """
        Generates points along a road segment and adds them to the grid.

        Parameters:
            road: A road segment object with necessary attributes.
        """
        if road.direction == -1:
            # Road is longer along the x-axis
            y_points = [
                road.y_length / 4 + road.y_min,
                3 * road.y_length / 4 + road.y_min,
            ]
            x_seg_length = road.x_length / np.floor(road.x_length / self.min_seg_length)
            x_points = (
                np.arange(x_seg_length / 2, road.x_length, x_seg_length) + road.x_min
            )
            for i in range(len(x_points) - 1):
                pt_A = Point(x_points[i], y_points[0])
                pt_C = Point(x_points[i + 1], y_points[0])
                if any(pt_A.within(ri) or pt_C.within(ri) for ri in self.road_intersections):
                    continue
                pt_B = Point(x_points[i], y_points[1])
                pt_D = Point(x_points[i + 1], y_points[1])
                self._add_vertex(pt_A, "east")
                self._add_vertex(pt_C, "east")
                self._add_vertex(pt_B, "west")
                self._add_vertex(pt_D, "west")
                self._add_points_road(pt_A, pt_B, pt_C, pt_D, road.direction)
        else:
            # Road is longer along the y-axis
            x_points = [
                road.x_length / 4 + road.x_min,
                3 * road.x_length / 4 + road.x_min,
            ]
            y_seg_length = road.y_length / np.floor(road.y_length / self.min_seg_length)
            y_points = (
                np.arange(y_seg_length / 2, road.y_length, y_seg_length) + road.y_min
            )
            for i in range(len(y_points) - 1):
                pt_A = Point(x_points[0], y_points[i])
                pt_C = Point(x_points[0], y_points[i + 1])
                if any(pt_A.within(ri) or pt_C.within(ri) for ri in self.road_intersections):
                    continue
                pt_B = Point(x_points[1], y_points[i])
                pt_D = Point(x_points[1], y_points[i + 1])
                self._add_vertex(pt_A, "south")
                self._add_vertex(pt_C, "south")
                self._add_vertex(pt_B, "north")
                self._add_vertex(pt_D, "north")
                self._add_points_road(pt_A, pt_B, pt_C, pt_D, road.direction)

    def _get_nearest_direction_neighbor(self, point: Point, kd_tree: KDTree, direction: str) -> Point:
        """
        Finds the nearest neighbor to a point in a specified direction.

        Parameters:
            point (Point): The point from which to find neighbors.
            kd_tree (KDTree): KDTree of existing points.
            direction (str): Direction to search ('north', 'south', 'east', 'west').

        Returns:
            Point: The nearest neighbor in the specified direction.

        Raises:
            ValueError: If an invalid direction is provided.
            Exception: If no neighbor is found in the specified direction.
        """
        _, indices = kd_tree.query(point.coords[0], k=20)
        if direction == "east":
            filtered_indices = [idx for idx in indices if self.vertices[idx][0].x < point.x]
        elif direction == "west":
            filtered_indices = [idx for idx in indices if self.vertices[idx][0].x > point.x]
        elif direction == "north":
            filtered_indices = [idx for idx in indices if self.vertices[idx][0].y > point.y]
        elif direction == "south":
            filtered_indices = [idx for idx in indices if self.vertices[idx][0].y < point.y]
        else:
            raise ValueError("Invalid direction! Use 'north', 'south', 'east', or 'west'.")
        if filtered_indices:
            nearest_idx = filtered_indices[0]
            return self.vertices[nearest_idx][0]
        else:
            raise Exception(f"Did not connect intersection point properly in {direction} direction.")

    def _generate_intersection_points(
        self,
        road_intersection: Polygon,
        road1,
        road2,
        road_pts_kd_tree: KDTree,
    ):
        """
        Generates points and edges for a road intersection.

        Parameters:
            road_intersection (Polygon): The polygon representing the intersection area.
            road1: The first road involved in the intersection.
            road2: The second road involved in the intersection.
            road_pts_kd_tree (KDTree): KDTree of existing road points.
        """
        # Determine which sides of the intersection have connecting roads
        directions = {'north': False, 'south': False, 'east': False, 'west': False}
        int_x_min, int_y_min, int_x_max, int_y_max = road_intersection.bounds
        if road1.y_max > int_y_max or road2.y_max > int_y_max:
            directions['north'] = True
        if road1.x_min < int_x_min or road2.x_min < int_x_min:
            directions['east'] = True
        if road1.y_min < int_y_min or road2.y_min < int_y_min:
            directions['south'] = True
        if road1.x_max > int_x_max or road2.x_max > int_x_max:
            directions['west'] = True

        # Create points on the intersection boundaries
        x_1 = (int_x_max - int_x_min) / 4 + int_x_min
        x_2 = 3 * (int_x_max - int_x_min) / 4 + int_x_min
        y_1 = (int_y_max - int_y_min) / 4 + int_y_min
        y_2 = 3 * (int_y_max - int_y_min) / 4 + int_y_min
        points = {
            'north': [Point(x_1, int_y_max), Point(x_2, int_y_max)],
            'south': [Point(x_1, int_y_min), Point(x_2, int_y_min)],
            'east': [Point(int_x_min, y_1), Point(int_x_min, y_2)],
            'west': [Point(int_x_max, y_1), Point(int_x_max, y_2)],
        }

        if directions['north']:
            self._add_vertex(points['north'][0], "south")
            self._add_vertex(points['north'][1], "north")
        if directions['south']:
            self._add_vertex(points['south'][0], "south")
            self._add_vertex(points['south'][1], "north")
        if directions['east']:
            self._add_vertex(points['east'][0], "east")
            self._add_vertex(points['east'][1], "west")
        if directions['west']:
            self._add_vertex(points['west'][0], "east")
            self._add_vertex(points['west'][1], "west")

        # Connect sides based on available directions
        self._connect_intersection_sides(directions, points)

        # Connect intersection points to the nearest road points
        for dir_key, is_present in directions.items():
            if is_present:
                self._add_points_road_intersection(
                    points[dir_key][0], points[dir_key][1], road_pts_kd_tree, dir_key
                )

    def _connect_intersection_sides(self, directions: Dict[str, bool], points: Dict[str, List[Point]]):
        """
        Connects the entrance and exit points of an intersection using Bezier curves or straight lines.

        Parameters:
            directions (Dict[str, bool]): Dictionary indicating which sides have connecting roads.
            points (Dict[str, List[Point]]): Dictionary of points for each side of the intersection.
        """
        # Define possible pairs of directions to connect
        connection_pairs = [
            ('north', 'south', 1),
            ('east', 'west', -1),
            ('north', 'east', 0),
            ('north', 'west', 0),
            ('south', 'east', 0),
            ('south', 'west', 0),
        ]

        for dir1, dir2, curve_direction in connection_pairs:
            if directions[dir1] and directions[dir2]:
                self._connect_intersection_points(
                    points[dir1], points[dir2], curve_direction
                )

    def _connect_intersection_points(self, side_1: List[Point], side_2: List[Point], direction: int):
        """
        Connects two sides of an intersection with edges.

        Parameters:
            side_1 (List[Point]): Points on the first side.
            side_2 (List[Point]): Points on the second side.
            direction (int): Direction indicator for Bezier curves.
        """
        if direction in (1, -1):
            # Straight connections
            self._add_edge(side_1[0], side_2[0], LineString([side_1[0], side_2[0]]))
            self._add_edge(side_1[1], side_2[1], LineString([side_1[1], side_2[1]]))
        else:
            # Bezier curve connections
            combinations = [(side_1[0], side_2[0]), (side_1[0], side_2[1]),
                            (side_1[1], side_2[1]), (side_1[1], side_2[0])]
            for pt1, pt2 in combinations:
                curve = self.bezier_curve(pt1, pt2, direction)
                self._add_edge(pt1, pt2, curve)

    def visualise(self, path=None, graph=None):
        """
        Visualizes the road network using matplotlib.

        Parameters:
            with_nodes (bool): If True, includes nodes and edges in the visualization.
        """
        if graph is None:
            graph = self.full_graph
        self.road_map.visualise(True, True, graph, path)

