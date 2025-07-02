import numpy as np
import cvxpy as cp
from shapely import Point, LineString
from typing import Tuple, List, Optional, Union
from copy import deepcopy
from statistics import mode

# from .path_planner import PathPointDatum
from .segment import *

PosePt2D = Tuple[float, float, float]  # (x, y, yaw) values


class MPCTracker:
    """
    Path tracker using MPC
    """

    def __init__(self, plot_rviz: bool = False) -> None:
        # time parameters:
        self.N = 5
        self.dt = 0.125  # follows tracker node rate

        # state limits
        self.v_max = 0.15  # m/s
        self.omega_max = 1  # rad/s
        self.delta_v_max = 0.1  # rate of change limit for linear speed
        self.delta_omega_max = 0.3  # rate of change limit for angular speed

        # cost function weights
        self.q_p = 8.0  # position error weight
        self.d_p = 1.0  # discount factor for position errors
        self.q_theta = 0.5  # orientation error weight
        self.d_theta = 1.0  # discount factor for orientation errors
        self.r_v = 1.0  # linear speed control effort
        self.r_omega = 1.0  # angular speed control effort
        self.r_v_smooth = 0.1  # smoothing linear velocity command
        self.r_omega_smooth = 0.1  # smoothing angular velocity command
        self.q_p_terminal = 10.0  # terminal error cost
        self.q_theta_terminal = 10.0  # terminal error cost
        self.t_p = 0  # transitioning position cost reduction

        # other parameters
        self.nominal_speed = self.v_max
        self.nominal_dl = self.dt * self.nominal_speed
        self.goal_radius = 0.03  # 30mm
        self.goal_angle_tol = 0.15 #0.0873 # rad or 5 deg

        # Progress variable
        self.s = None  # percentage progress along the path
        self.path_length = None  # total length of the path

        # previous values for warm starts
        self.x_prev = None
        self.y_prev = None
        self.theta_prev = None
        self.v_prev = None
        self.omega_prev = None

        # declare other variables
        self.transitioning = False
        self.final_theta = None

        # write
        self.write = False
        if self.write:
            with open("src/mpc_data.txt", "w") as file:
                file.write(f"Total error, OE, current theta, des OE, optimised OE \n")

        # rviz
        self.plot_rviz = plot_rviz

    def compute_path_length(self, path: List[Segment]) -> None:
        """
        Computes cumulative distances along the path and stores the total path length.

        Args:
            path(List[Segment]): List of desired positions, with direction of motion
        """
        self.path_length = 0
        for seg in path:
            self.path_length += seg.length

    def is_at_goal_position(self, current_pose: Pose2D, path: List[Segment]) -> bool:
        goal = path[-1].end
        goal_distance_sq = (current_pose[0] - goal.x) ** 2 + (current_pose[1] - goal.y) ** 2
        return goal_distance_sq < self.goal_radius ** 2 and self.s is not None and self.s > 0.5

    def compute_final_theta(self, path: List[Segment]) -> None:
        """
        Computes final theta for when path is too short to compute the angle.

        Args:
            path(List[Segment]): List of desired positions, with direction of motion
        """
        dy = path[-1].end.y - path[-1].start.y
        dx = path[-1].end.x - path[-1].start.x
        self.final_theta = np.arctan2(dy, dx)
        if path[-1].direction == -1:
            # Correctly adjust by adding π radians
            self.final_theta += np.pi
        #
        # Normalize the angle to [-π, π]
        self.final_theta = (self.final_theta + np.pi) % (2 * np.pi) - np.pi
        #
        theta_a = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi) - np.pi
        theta_b = np.arctan2(dy, dx) + np.pi
        theta_b = (theta_b + np.pi) % (2 * np.pi) - np.pi

    def find_segment_and_progress(
        self, path: List[Segment], path_travelled: float
    ) -> Tuple[int, float]:
        """Find the segment and progress along it based on total path traveled.

        Args:
            path (List[Segment]): List of path segments.
            path_travelled (float): Distance travelled along the path.

        Returns:
            Tuple[int, float]: A tuple containing:
                - int: Index of the segment where the distance falls.
                - float: Progress percentage along the identified segment.
        """
        path_travelled_tmp = 0
        for idx, seg in enumerate(path):
            path_travelled_tmp += seg.length
            if seg.length == 0:
                continue
            if path_travelled_tmp >= path_travelled:
                percent_travelled = 1 - (path_travelled_tmp - path_travelled) / seg.length
                return idx, percent_travelled
        return len(path) - 1, 1  # If path_traveled exceeds path length, return last segment

    def adjust_start_position(
        self, path: List[Segment], segment_index: int, percent_travelled: float
    ) -> Segment:
        """Adjust the start position of a segment based on travel progress.

        Args:
            path (List[Segment]): List of path segments.
            segment_index (int): Index of the current segment.
            percent_travelled (float): Percentage of progress within the segment.

        Returns:
            Segment: A modified segment with the adjusted start position.
        """
        seg = deepcopy(path[segment_index])
        seg.start.x = (path[segment_index].end.x - path[segment_index].start.x) * percent_travelled + path[segment_index].start.x
        seg.start.y = (path[segment_index].end.y - path[segment_index].start.y) * percent_travelled + path[segment_index].start.y
        seg.re_init()
        return seg
    
    def calculate_progress_on_segment(self, nearest_pt: Point, segment: Segment) -> float:
        """Calculate the percentage along the full segment where the nearest point lies.

        Args:
            nearest_pt (Point): The closest point on the segment to the robot.
            segment (Segment): The full path segment.

        Returns:
            float: The accurate percentage along the segment.
        """
        dx = segment.end.x - segment.start.x
        dy = segment.end.y - segment.start.y
        px = nearest_pt.x - segment.start.x
        py = nearest_pt.y - segment.start.y

        # Calculate percentage using dot product
        if (dx**2 + dy**2) == 0:
            return 1.0  # Segment has zero length
        else:
            percent = (px * dx + py * dy) / (dx**2 + dy**2)
            # Clamp to ensure it remains within [0, 1]
            return max(0.0, min(1.0, percent))

    def find_closest_point_on_remaining_path(
        self, current_pose: PosePt2D, remaining_path: List[Segment], full_path: List[Segment], start_segment_index: int
    ) -> Tuple[int, float, Tuple[float, float]]:
        """Find the closest point to the robot on the remaining path with reference to the full path.

        Args:
            current_pose (PosePt2D): Current position of the robot.
            remaining_path (List[Segment]): Remaining path segments starting from the adjusted position.
            full_path (List[Segment]): Full original path segments.
            start_segment_index (int): The index in the original full path where this subset starts.

        Returns:
            Tuple[int, float, Tuple[float, float]]: A tuple containing:
                - int: Index of the nearest segment in the original path.
                - float: Accurate progress percentage along the segment in the full path.
                - Tuple[float, float]: (x, y) coordinates of the nearest point.
        """
        min_distance = float("inf")
        robot_position = Point(current_pose[0], current_pose[1])
        nearest_pt, nearest_seg_idx_in_rem_path, percent_of_nearest_segment = None, None, None

        # Initial pass to find the nearest segment in `remaining_path`
        for idx, seg in enumerate(remaining_path):
            # Project and normalize
            projected_dist = seg.line.project(robot_position)
            nearest_pt_tmp = seg.line.interpolate(projected_dist)
            distance = robot_position.distance(nearest_pt_tmp)
            
            if distance < min_distance:
                min_distance = distance
                nearest_pt = nearest_pt_tmp
                nearest_seg_idx_in_rem_path = idx

        # Map to the correct segment index in `full_path`
        nearest_seg_idx = nearest_seg_idx_in_rem_path + start_segment_index

        # Compute accurate percentage on the segment in `full_path`
        percent_of_nearest_segment = self.calculate_progress_on_segment(nearest_pt, full_path[nearest_seg_idx])
        # Handle edge case where the nearest point is near the end of a segment or segment has zero length
        # for smoothness
        while (percent_of_nearest_segment >= 0.995 or remaining_path[nearest_seg_idx_in_rem_path].length == 0):
            # Check if we are on the last segment in `remaining_path`, in which case we stop
            if nearest_seg_idx_in_rem_path == len(remaining_path) - 1:
                break  # Exit if we’re at the last segment to prevent infinite loop

            # Move to the next segment if possible
            nearest_seg_idx_in_rem_path += 1
            nearest_seg_idx = nearest_seg_idx_in_rem_path + start_segment_index
            if remaining_path[nearest_seg_idx_in_rem_path].length > 0:
                nearest_pt = remaining_path[nearest_seg_idx_in_rem_path].start
                percent_of_nearest_segment = 0
                break  # Exit as soon as a valid non-zero-length segment is found

        return nearest_seg_idx, percent_of_nearest_segment, (nearest_pt.x, nearest_pt.y)

    def find_nearest_point_on_path(
        self, current_pose: PosePt2D, path: List[Segment], s: float
    ) -> Tuple[int, float, Tuple[float, float]]:
        """Main function to find the closest point on a path.

        Args:
            current_pose (PosePt2D): Current position of the robot.
            path (List[Segment]): List of desired path segments.
            s (float): Progress percentage along the path.

        Returns:
            Tuple[int, float, Tuple[float, float]]: A tuple containing:
                - int: Index of the nearest segment.
                - float: Percentage along the segment.
                - Tuple[float, float]: (x, y) coordinates of the nearest point.
        """
        path_travelled = self.path_length * s
        segment_idx, percent_travelled = self.find_segment_and_progress(path, path_travelled)
        adjusted_starting_segment = self.adjust_start_position(path, segment_idx, percent_travelled)
        if segment_idx + 1 < len(path):
            remaining_path = [adjusted_starting_segment] + path[segment_idx + 1 :]
        else:
            remaining_path = [adjusted_starting_segment]
        return self.find_closest_point_on_remaining_path(current_pose, remaining_path, path, segment_idx)

    def update_progress_variable(
        self,
        path: List[Segment],
        nearest_seg_idx: int,
        percent_of_nearest_segment: float,
    ) -> float:
        """
        Updates or computes progress variable s, which contains the percentage the robot is along the path

        Args:
            path(List[Segment]): List of desired positions, with direction of motion
            nearest_seg_idx(int): segment index closest to the current position
            percent_of_nearest_segment(float): percentage distance along the segment to the nearest point to the robot

        Returns:
            (float): updated path progress variable
        """
        length_travelled = 0
        for seg in path[0:nearest_seg_idx]:
            length_travelled += seg.length
        length_travelled += path[nearest_seg_idx].length * percent_of_nearest_segment
        # print(self.s, flush=True)
        new_s = length_travelled / self.path_length
        if self.s is not None and new_s - self.s < -0.0001:
            print(f"WARNING: s decreasing from {self.s} to {new_s}", flush=True)
        return new_s

    def get_remaining_path(
        self, current_pose: PosePt2D, path: List[Segment]
    ) -> List[Segment]:
        """
        Updates or computes progress variable s, which contains the percentage the robot is along the path

        Args:
            current_pose(PosePt2D): Current position of the robot
            path(List[Segment]): List of desired positions, with direction of motion
        Returns:
            List[Segment]: Path remaining after the current pose, and not before path progress
        """
        if self.s is None:
            # get nearest position to the path, and compute the value s
            nearest_seg_idx, percent_of_nearest_segment, nearest_point = (
                self.find_nearest_point_on_path(current_pose, path, 0.0)
            )
        else:
            # restrict path to s onward, and then compute nearest position to path
            nearest_seg_idx, percent_of_nearest_segment, nearest_point = (
                self.find_nearest_point_on_path(current_pose, path, self.s)
            )
        self.s = self.update_progress_variable(
            path, nearest_seg_idx, percent_of_nearest_segment
        )
        path_seg_tmp = deepcopy(path[nearest_seg_idx])
        path_seg_tmp.start.x = nearest_point[0]
        path_seg_tmp.start.y = nearest_point[1]
        path_seg_tmp.re_init()
        if nearest_seg_idx + 1 < len(path):
            remaining_path = [path_seg_tmp] + path[nearest_seg_idx + 1 :]
        else:
            remaining_path = [path_seg_tmp]
        return remaining_path

    def calculate_theta(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        current_yaw: float,
        direction: List[int],
    ) -> np.ndarray[float]:
        """
        Computes the angle between the current position and the next positon.
        If direction is 0 (stop waypoint), either follow the previous or following theta.

        Args:
            x(np.ndarray[float]): x coordinates
            y(np.ndarray[float]): y coordinates
            current_yaw(float): Current yaw of the robot
            direction(List[int]): direction, either 1 (forward), -1 (backwards) or 0 (stop)
        Returns:
            (np.ndarray[float]): Array of thetas
        """
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)
        theta = np.zeros(len(x))
        theta[0] = current_yaw
        for i in range(1, len(theta)):
            if dx[i - 1] == 0 and dy[i - 1] == 0:
                desired_theta = self.final_theta if self.final_theta is not None else theta[i - 1]
            elif direction[i - 1] != direction[i]:
                desired_theta = theta[i - 1]
            elif direction[i - 1] == 1:
                desired_theta = angles[i - 1]
            elif direction[i - 1] == -1:
                desired_theta = angles[i - 1] + np.pi
            theta[i] = desired_theta
        theta = (theta - current_yaw + np.pi) % (2 * np.pi) - np.pi
        return theta

    def get_next_reference_point(
        self, remaining_path: list[Segment]
    ) -> Tuple[float, float, int, list[Segment]]:
        """
        Finds the next look ahead point based on the nominal length step

        Args:
            remaining_path (list[Segment]): List of desired poses
        Returns:
            (Tuple[float, float, int, list[Segment]]): (x, y, direction) look ahead point of
            nominal length step distance from the start of the desired path if the direction
            is constant.
            Otherwise, the look ahead point is where the direction has changed.
            Also, returns the path remaining, starting at the identified look ahead point.
        """
        remaining_dl = self.nominal_dl
        for seg_idx, seg in enumerate(remaining_path):
            if remaining_dl <= seg.length:
                next_x = seg.start.x + remaining_dl / seg.length * (
                    seg.end.x - seg.start.x
                )
                next_y = seg.start.y + remaining_dl / seg.length * (
                    seg.end.y - seg.start.y
                )
                next_dir = seg.direction
                tmp_seg = deepcopy(seg)
                tmp_seg.start.x = next_x
                tmp_seg.start.y = next_y
                tmp_seg.re_init()
                if seg_idx + 1 < len(remaining_path):
                    remaining_path = [tmp_seg] + remaining_path[seg_idx + 1 :]
                else:
                    remaining_path = [tmp_seg]
                return next_x, next_y, next_dir, remaining_path
            else:
                remaining_dl -= seg.length
                if seg_idx == len(remaining_path) - 1:
                    # If we've reached the end of the path
                    next_x = seg.end.x
                    next_y = seg.end.y
                    next_dir = seg.direction
                    tmp_seg = deepcopy(seg)
                    tmp_seg.start = seg.end
                    tmp_seg.re_init()
                    remaining_path = [tmp_seg]
                    return next_x, next_y, next_dir, remaining_path
                else:
                    continue

    def get_reference_path(
        self, current_pose: PosePt2D, path: List[Segment]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Finds where the robot is closest to the path, interpolates the next N poses.

        Args:
            current_pose (PosePt2D): Current position of the robot
            path (List[Segment]): List of desired positions, with direction of motion
        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]): Desired x, desired y,
            desired yaw and desired direction of the robot of the next N timesteps.
        """
        if self.path_length is None:
            self.compute_path_length(path)
        if self.final_theta is None:
            self.compute_final_theta(path)
        remaining_spatial_path = self.get_remaining_path(current_pose, path)
        x_ref = np.zeros(self.N + 1)
        y_ref = np.zeros(self.N + 1)
        theta_ref = np.zeros(self.N + 1)
        direction_ref = [0] * (self.N + 1)

        x_ref[0], y_ref[0] = current_pose[0], current_pose[1]
        for k in range(1, self.N + 1):
            x_ref[k], y_ref[k], direction_ref[k], remaining_spatial_path = (
                self.get_next_reference_point(remaining_spatial_path)
            )
        direction_ref[0] = direction_ref[1]
        theta_ref = self.calculate_theta(x_ref, y_ref, current_pose[2], direction_ref)
        return x_ref, y_ref, theta_ref, direction_ref

    def warm_start_shift(self, arr: np.ndarray) -> np.ndarray:
        """
        Shifts all values down by one for warm start

        Args:
            arr (np.ndarray): previous calculated decision variable
        Returns:
            (np.ndarray): shifted array
        """
        return np.concatenate([arr[1:], [arr[-1]]])

    def initialise_new_path(self) -> None:
        """
        Informs tracker there is a new path, and resets path progress variables.
        """
        self.s = None
        self.path_length = None
        self.final_theta = None

    def check_goal(self, current_pose: Pose2D, path: List[Segment], return_angle: bool = False) -> Union[
        bool, Tuple[bool, float]]:
        goal = path[-1].end
        tmp_pt = path[-1].start
        if tmp_pt == goal and len(path) > 1:
            tmp_pt = path[-2].start
        goal_distance_sq = (current_pose[0] - goal.x) ** 2 + (current_pose[1] - goal.y) ** 2

        if self.final_theta is None:
            self.compute_final_theta(path)
        goal_angle = self.final_theta

        angle_diff = np.abs((goal_angle - current_pose[2] + np.pi) % (2 * np.pi) - np.pi)
        s_str = f"{self.s:.2f}" if self.s is not None else "None"
        print(f"[MPC] dist²={goal_distance_sq:.4f}  angle_diff={angle_diff:.4f}  s={s_str}", flush=True)

        goal_reached = (
                (goal_distance_sq < self.goal_radius ** 2) and
                (angle_diff < self.goal_angle_tol) and
                self.s is not None and self.s > 0.5
        )

        if return_angle:
            return goal_reached, angle_diff
        else:
            return goal_reached

    def track(self, current_pose: PosePt2D, path: List[Segment]) -> Tuple[float, float]:
        """
        Tracks the path of the robot using an MPC solver

        Args:
            current_pose (PosePt2D): Current position of the robot
            path (List[Segment]): List of desired positions, with direction of motion
        Returns:
            Tuple[float, float]: command linear and angular velocity
        """
        if self.s is not None and self.check_goal(current_pose, path):
            # print("MPC goal reached!", flush=True)
            if self.plot_rviz:
                return 0.0, 0.0, [], [], []
            else:
                return 0.0, 0.0
        # path = self.remove_zero_length_segments(path)
        x_ref, y_ref, theta_ref, direction_ref = self.get_reference_path(
            current_pose, path
        )
        # self.transitioning = direction_ref[0] != direction_ref[1]  # Check for transition
        # print(path[0:4], flush=True)
        # print("x ", x_ref, flush=True)
        # print("y ", y_ref, flush=True)
        # print("oe ", theta_ref, flush=True)
        # print("dir ", direction_ref, flush=True)

        # state var
        x = cp.Variable(self.N + 1)
        y = cp.Variable(self.N + 1)
        # here, theta is not the angle in the world frame but it is the orientation error between the path and the robot
        theta = cp.Variable(self.N + 1)

        # control var
        v = cp.Variable(self.N)
        omega = cp.Variable(self.N)

        # warm start
        if self.x_prev is not None:
            x.value = self.x_prev
        if self.y_prev is not None:
            y.value = self.y_prev
        if self.theta_prev is not None:
            theta.value = self.theta_prev
        if self.v_prev is not None:
            v.value = self.v_prev
        if self.omega_prev is not None:
            omega.value = self.omega_prev

        # define objective function
        position_error = 0
        orientation_error = 0
        control_effort = 0
        control_smoothness = 0

        objective = 0
        constraints = []

        for k in range(self.N + 1):
            position_error += (
                    self.d_p**k * self.q_p * (x[k] - x_ref[k]) ** 2
                    + self.d_p**k * self.q_p * (y[k] - y_ref[k]) ** 2
                )
            # if not self.transitioning or (self.transitioning and direction_ref[k] == direction_ref[-1]):
            #     position_error += (
            #         self.d_p**k * self.q_p * (x[k] - x_ref[k]) ** 2
            #         + self.d_p**k * self.q_p * (y[k] - y_ref[k]) ** 2
            #     )
            # else:
            #     position_error += (
            #         self.t_p * self.d_p**k * self.q_p * (x[k] - x_ref[k]) ** 2
            #         + self.t_p * self.d_p**k * self.q_p * (y[k] - y_ref[k]) ** 2
            #     )

            orientation_error += (
                self.d_theta**k * self.q_theta * (theta[k] - theta_ref[k]) ** 2
            )

            if k == self.N:
                position_error += self.q_p_terminal * (
                    cp.square(x[k] - x_ref[k]) + cp.square(y[k] - y_ref[k])
                )
                orientation_error += self.q_theta_terminal * (
                    (theta[k] - theta_ref[k]) ** 2
                )

        control_effort += self.r_v * cp.sum_squares(v) + self.r_omega * cp.sum_squares(
            omega
        )

        for k in range(self.N - 1):
            control_smoothness += self.r_v_smooth * cp.square(v[k + 1] - v[k])
            control_smoothness += self.r_omega_smooth * cp.square(
                omega[k + 1] - omega[k]
            )

        objective = (
            position_error + orientation_error + control_effort + control_smoothness
        )

        # initial conditions
        initial_conditions = (
            [x[0] == current_pose[0]]
            + [y[0] == current_pose[1]]
            + [theta[0] == 0]  # since theta is the orientation error
        )

        # system dynamics
        system_dynamics = []
        for k in range(self.N):
            cos_theta_ref = np.cos(theta_ref[k] + current_pose[2])
            sin_theta_ref = np.sin(theta_ref[k] + current_pose[2])
            system_dynamics += [x[k + 1] == x[k] + v[k] * cos_theta_ref * self.dt]
            system_dynamics += [y[k + 1] == y[k] + v[k] * sin_theta_ref * self.dt]
            system_dynamics += [theta[k + 1] == theta[k] + omega[k] * self.dt]

        # velocity constraints
        velocity_limits = []
        velocity_limits += [cp.abs(v) <= self.v_max]
        velocity_limits += [cp.abs(omega) <= self.omega_max]

        # direction constraints
        direction_constraints = []
        # if direction_ref[k] == 1:  # forward
        #     if self.transitioning:  # Relaxed constraint
        #         direction_constraints += [v[k] >= -0.5 * self.v_max]
        #     else:
        #         direction_constraints += [v[k] >= -0.1*self.v_max]
        # elif direction_ref[k] == -1:  # backward
        #     if self.transitioning:  # Relaxed constraint
        #         direction_constraints += [v[k] <= 0.5 * self.v_max]
        #     else:
        #         direction_constraints += [v[k] <= 0.1*self.v_max]
        # elif direction_ref[k] == 0:  # stop
        #     direction_constraints += [cp.abs(v[k]) <= self.v_max * 0.1]


        for k in range(self.N):
            if direction_ref[k] == 1:  # forward
                direction_constraints += [v[k] >= -0.1*self.v_max]
            elif direction_ref[k] == -1:  # backward
                direction_constraints += [v[k] <= 0.1*self.v_max]
            elif direction_ref[k] == 0:  # stop
                direction_constraints += [cp.abs(v[k]) <= self.v_max * 0.1]

        # rate of change constraints
        roc_constraints = []
        roc_constraints += [
            cp.abs(v[k + 1] - v[k]) <= self.delta_v_max for k in range(self.N - 1)
        ]
        roc_constraints += [
            cp.abs(omega[k + 1] - omega[k]) <= self.delta_omega_max
            for k in range(self.N - 1)
        ]

        constraints += (
            initial_conditions
            + system_dynamics
            + velocity_limits
            + direction_constraints
            + roc_constraints
        )

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True)

        if prob.status != cp.OPTIMAL:
            # TODO: We still need it to return something
            raise ValueError("Solver did not find an optimal solution.")

        # print("Objective cost:", prob.value, flush=True)
        # print("Position error cost:", position_error.value)
        # print("Orientation error cost:", orientation_error.value)
        # print("Control effort cost:", control_effort.value)
        # print("Control smoothness cost:", control_smoothness.value)

        v_command = v.value[0]
        omega_command = omega.value[0]

        # self.x_prev = self.warm_start_shift(x.value)
        # self.y_prev = self.warm_start_shift(y.value)
        # self.theta_prev = self.warm_start_shift(theta.value)
        # self.v_prev = self.warm_start_shift(v.value)
        # self.omega_prev = self.warm_start_shift(omega.value)

        # print("Velocity: ",v.value, flush=True)

        if np.abs(v_command) < 0.001 * self.v_max:
            # add a kick
            v_command = direction_ref[0] * 0.9 * self.v_max
            omega_command = (np.random.random() - 0.5) * 2

        if self.write:
            with open("src/mpc_data.txt", "a") as file:
                file.write(
                    f"{prob.value:.5f}, {orientation_error.value:.5f}, {current_pose[2]:.4f}, {theta_ref[1]:.4f}, {theta.value[1]:.4f} \n"
                )
        if self.plot_rviz:
            return v_command, omega_command, x.value, y.value, theta.value

        return v_command, omega_command
