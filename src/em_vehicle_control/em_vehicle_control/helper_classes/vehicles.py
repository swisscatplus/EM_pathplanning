from __future__ import annotations

from typing import Tuple, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import Polygon, Point, box, MultiPolygon
try:
    from shapely.geometry.base import BaseGeometry  # Shapely 1.x / 2.x compatible spot
except ImportError:
    BaseGeometry = object  # fallback for type hints only
from shapely.ops import unary_union
from shapely.affinity import rotate, translate


class BaseVehicle:
    """
    Base class for differential-drive style robots.
    θ convention: measured **clockwise** from +X (to match your original comment).
    Many math libs (Shapely rotations, standard robotics) use CCW; we compensate when needed.
    """

    def __init__(
        self,
        start_position: Tuple[float, float, float],
        wheel_radius: float,
        length_of_wheel_axis: float,
    ):
        self._x = float(start_position[0])
        self._y = float(start_position[1])
        self._theta = float(start_position[2])  # clockwise from +X (see note above)

        self._wheel_radius = float(wheel_radius)
        self._length_of_wheel_axis = float(length_of_wheel_axis)

        self._L_wheel_speed = 0.0
        self._R_wheel_speed = 0.0
        self._vehicle_model: Optional[BaseGeometry] = None
        self._linear_velocity: Optional[float] = None
        self._angular_velocity: Optional[float] = None

        # Public alias if you want to access it elsewhere
        self.vehicle_model: Optional[BaseGeometry] = None

    def visualise(self) -> None:
        """Plot the current vehicle footprint (Polygon or MultiPolygon)."""
        if self._vehicle_model is None:
            raise RuntimeError("Vehicle geometry is not constructed yet.")
        geoms: List[Polygon] = []
        if isinstance(self._vehicle_model, Polygon):
            geoms = [self._vehicle_model]
        elif isinstance(self._vehicle_model, MultiPolygon):
            geoms = list(self._vehicle_model.geoms)
        else:
            raise TypeError(f"Unsupported geometry type: {type(self._vehicle_model)}")

        fig, ax = plt.subplots()
        for poly in geoms:
            x, y = poly.exterior.xy
            patch = MplPolygon(list(zip(x, y)), closed=True,
                               edgecolor="black", facecolor="lightblue", alpha=0.5)
            ax.add_patch(patch)

        ax.set_aspect("equal", adjustable="box")
        ax.autoscale()
        ax.grid(True, alpha=0.3)
        plt.show()

    # If you want dynamics, uncomment and use:
    # def step(self, dt: float, left_acc: float, right_acc: float) -> None:
    #     """Updates the state after dt with inputs left_acc/right_acc (rad/s^2)."""
    #     self._L_wheel_speed += float(left_acc) * dt
    #     self._R_wheel_speed += float(right_acc) * dt
    #     # Linear and angular velocities (differential drive kinematics)
    #     self._linear_velocity = self._wheel_radius * (self._L_wheel_speed + self._R_wheel_speed) / 2.0
    #     self._angular_velocity = self._wheel_radius * (self._R_wheel_speed - self._L_wheel_speed) / self._length_of_wheel_axis
    #     # Integrate pose: using your θ clockwise convention → cos/sin still fine for translation
    #     self._x += self._linear_velocity * dt * np.cos(self._theta)
    #     self._y += self._linear_velocity * dt * np.sin(self._theta)
    #     self._theta = (self._theta + self._angular_velocity * dt) % (2.0 * np.pi)

    def get_state(self) -> Tuple[float, float, float, Optional[float], Optional[float]]:
        return self._x, self._y, self._theta, self._linear_velocity, self._angular_velocity


class EdyMobile(BaseVehicle):
    """
    EdyMobile footprint = rectangle (width = wheel axle length, length = back+front)
    unioned with a circular "nose" at the front. Robust construction using
    shapely's box/rotate/translate to avoid invalid polygons.
    """
    def __init__(self, start_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        # Dimensions (meters)
        self._length_of_wheel_axis = 0.131  # axle length → robot width
        self._length_to_back_rect = 0.03
        self._length_to_front_rect = 0.11

        # Wheel and motion parameters
        self._wheel_radius = 0.07
        self.turning_radius = self._length_of_wheel_axis / 2.0
        self.nominal_speed = 0.3  # m/s

        super().__init__(
            start_position=start_position,
            wheel_radius=self._wheel_radius,
            length_of_wheel_axis=self._length_of_wheel_axis,
        )

        self.construct_vehicle()

    def construct_vehicle(self, state: Optional[Tuple[float, float, float]] = None):
        """Rebuild the shapely geometry representing the vehicle at the given (or current) state."""
        if state is not None:
            self._x, self._y, self._theta = float(state[0]), float(state[1]), float(state[2])

        vals = np.array([
            self._x, self._y, self._theta,
            self._length_of_wheel_axis, self._length_to_back_rect, self._length_to_front_rect
        ], dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"EdyMobile.construct_vehicle got non-finite values: {vals}")

        # Rectangle dimensions
        L = float(self._length_to_front_rect + self._length_to_back_rect)  # along heading
        W = float(self._length_of_wheel_axis)  # across heading

        if L <= 0.0 or W <= 0.0:
            raise ValueError(f"Invalid EdyMobile dims: L={L}, W={W}")

        # 1) Base rectangle centered at origin (valid/closed polygon)
        rect = box(-L / 2.0, -W / 2.0, L / 2.0, W / 2.0)

        # 2) Rotate by yaw; Shapely's rotation is CCW, but θ here is clockwise → negate
        rect = rotate(rect, -self._theta, origin=(0.0, 0.0), use_radians=True)

        # 3) Translate so that the rectangle's center sits between back/front offsets
        center_forward = (self._length_to_front_rect - self._length_to_back_rect) / 2.0
        dx = center_forward * np.cos(self._theta)
        dy = center_forward * np.sin(self._theta)
        rect = translate(rect, xoff=self._x + dx, yoff=self._y + dy)

        # 4) Front circle (nose) centered at the front offset along heading
        nose_radius = self._length_of_wheel_axis / 2.0
        fx = self._x + self._length_to_front_rect * np.cos(self._theta)
        fy = self._y + self._length_to_front_rect * np.sin(self._theta)
        circle = Point(fx, fy).buffer(nose_radius, resolution=64)

        # 5) Union & clean
        shape = unary_union([rect, circle]).buffer(0)

        self._vehicle_model = shape
        self.vehicle_model = self._vehicle_model

    # If you enable BaseVehicle.step(), you can add:
    # def step(self, dt: float, left_acc: float, right_acc: float) -> None:
    #     super().step(dt, left_acc, right_acc)
    #     self.construct_vehicle()


class Edison(BaseVehicle):
    """
    Edison: simple circular body robot.
    """
    def __init__(self, start_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self._length_of_wheel_axis = 0.23
        self.body_radius = 0.245 / 2.0
        self._wheel_radius = 0.08
        self.turning_radius = self._length_of_wheel_axis / 2.0
        self.nominal_speed = 0.3  # m/s

        super().__init__(
            start_position=start_position,
            wheel_radius=self._wheel_radius,
            length_of_wheel_axis=self._length_of_wheel_axis,
        )
        self.construct_vehicle()

    def construct_vehicle(self, state: Optional[Tuple[float, float, float]] = None):
        if state is not None:
            self._x, self._y, self._theta = float(state[0]), float(state[1]), float(state[2])

        vals = np.array([self._x, self._y, self.body_radius], dtype=float)
        if not np.isfinite(vals).all() or self.body_radius <= 0.0:
            raise ValueError(f"Edison.construct_vehicle bad inputs: {vals}")

        circle = Point(self._x, self._y).buffer(self.body_radius, resolution=64).buffer(0)
        self._vehicle_model = circle
        self.vehicle_model = self._vehicle_model

    # def step(self, dt: float, left_acc: float, right_acc: float) -> None:
    #     super().step(dt, left_acc, right_acc)
    #     self.construct_vehicle()
