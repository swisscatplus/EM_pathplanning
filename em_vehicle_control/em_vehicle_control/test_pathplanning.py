from helper_classes.map import RoadMap, RoadSegment, RoadGraph, RoadTrack
from helper_classes.path_planner import PathPlanner
from helper_classes.vehicles import EdyMobile, Edison
from helper_classes.pathplanners.rrt import *
from helper_classes.pathplanners.pp_viz import RoadMapAnimator
from helper_classes.pathplanners.coopAstar import CAstar
from shapely import within
import time
import networkx as nx

test_roads = [
    RoadSegment((2.87, 1.67), (3.52, 13.67)),
    RoadSegment((6.9, 0.3), (7.55, 15.67)),
    RoadSegment((4.72, 1.67), (5.7, 8.64)),
    RoadSegment((2.87, 8), (7.55, 8.64)),
    RoadSegment((2.87, 1.67), (7.55, 2.32)),
]
test_map = RoadMap(test_roads)
# vehicle = EdyMobile(start_position=(3.218,3,np.pi/2))
# goal = (3.218,3.3,-np.pi/2)
# goal_radius = 0.2
# rrt = RRT_star_Reeds_Shepp((3.218,3,np.pi/2), goal, goal_radius, vehicle, test_map.map,
#             max_iter=100, search_until_max_iter=False, visualise=True)
test_graph = RoadTrack(test_map)
# test_graph.visualise()
print(test_graph.full_graph.nodes[148]['pos'])

def dist(a, b):

    (x1, y1) = a

    (x2, y2) = b

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

path = nx.astar_path(test_graph.full_graph, 0, 118, heuristic=None, weight='weight')
# test_graph.visualise(path)
from shapely.geometry import LineString, Point

line = LineString([(0,0),(1,0)])
# point = Point(2, 99)
# print(line.project(point, True))

# test_pp = PathPlanner(test_map, test_graph, 3)
# test_graph.visualise()
# test_rrt_start_pose = (3.145, 1.875, -3 * np.pi / 4)
# test_rrt_end_pose = (5.0, 2, np.pi / 2)
# test_ca_pose = (3.58, 1.832, 0)
# test_goal = (5.204, 1.944, np.pi / 2)