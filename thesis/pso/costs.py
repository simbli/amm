"""
Contains the CostFunction class for PSO objectives optimization.
"""

import numpy as np
from shapely import geometry, ops


class CostFunction:
    """
    Implements weighted costs for waypoints planning and obstacle avoidance.
    """

    fuel_rate_pto = 1.5
    fuel_rate_mec = 2.5
    fuel_rate_pti = 2.0
    segment_length = 100
    c_segment = 10**-8

    def __init__(
        self,
        version: int,
        obstacles: geometry.MultiPolygon,
        winds: list[float, float],
    ):
        match version:
            case 10:
                self.zeta = 0.01
                self.c_grounding = 40
                self.c_path = 10**-4
                self.objectives = self.objectives_f10
            case 12:
                self.zeta = 0.04
                self.c_grounding = 10000
                self.c_path = 10**-4
                self.c_wind = 100
                self.objectives = self.objectives_f12
            case 16:
                self.zeta = 0.04
                self.c_mso = 10**-4
                self.c_path = 10**-4
                self.c_wind = 1
                self.prob_scale = 5 * 10**10
                self.objectives = self.objectives_f16
            case _:
                raise ValueError
        self.version = version
        self.obstacles = obstacles
        self.wind_direction = winds[0] * np.pi / 180
        self.wind_velocity = winds[1]
        self.d_x = np.sin(self.wind_direction)
        self.d_y = np.cos(self.wind_direction)

    def objectives_f10(
        self,
        _,
        xy: tuple[float, float],
        wp_ref0: geometry.Point,
        wp_ref1: geometry.Point,
        wp_ref2: geometry.Point,
    ) -> dict:
        point = geometry.Point(xy)
        return dict(
            costs=(
                self.c_path * self._path_deviation_cost(point, wp_ref0),
                self.c_segment * self._segment_cost(point, wp_ref1, wp_ref2),
                self.c_grounding * self._obstacle_cost(point),
            )
        )

    def objectives_f12(
        self,
        _,
        xy: tuple[float, float],
        wp_ref0: geometry.Point,
        wp_ref1: geometry.Point,
        wp_ref2: geometry.Point,
    ) -> dict:
        point = geometry.Point(xy)
        return dict(
            costs=(
                self.c_path * self._path_deviation_cost(point, wp_ref0),
                self.c_segment * self._segment_cost(point, wp_ref1, wp_ref2),
                self.c_grounding * self._obstacle_cost(point),
                self.c_wind * self._wind_cost(point),
            )
        )

    def objectives_f16(
        self,
        probs: dict,
        xy: tuple[float, float],
        wp_ref0: geometry.Point,
        wp_ref1: geometry.Point,
        wp_ref2: geometry.Point,
    ) -> dict:
        point = geometry.Point(xy)
        mso_cost, mso, details = self._mso_mode_costs(probs, point, wp_ref0)
        return dict(
            costs=(
                self.c_path * self._path_deviation_cost(point, wp_ref0),
                self.c_segment * self._segment_cost(point, wp_ref1, wp_ref2),
                self.c_wind * self._wind_cost(point),
                self.c_mso * mso_cost,
            ),
            mso=mso,
            mso_costs=details,
        )

    def _mso_mode_costs(
        self,
        probs: dict,
        location: geometry.Point,
        waypoint_reference: geometry.Point,
    ) -> tuple:
        distance = location.distance(waypoint_reference)
        distance_weight = self.segment_length + distance
        fuel_pto = self.fuel_rate_pto * distance_weight
        fuel_mec = self.fuel_rate_mec * distance_weight
        fuel_pti = self.fuel_rate_pti * distance_weight
        grounding_pto = probs['pto'] * self.prob_scale
        grounding_mec = probs['mec'] * self.prob_scale
        grounding_pti = probs['pti'] * self.prob_scale
        cost_pto = grounding_pto + fuel_pto
        cost_mec = grounding_mec + fuel_mec
        cost_pti = grounding_pti + fuel_pti
        if cost_pto <= cost_mec and cost_pto <= cost_pti:
            mso, cost = 'pto', cost_pto
        elif cost_mec <= cost_pti:
            mso, cost = 'mec', cost_mec
        else:
            mso, cost = 'pti', cost_pti
        details = [
            fuel_pto,
            grounding_pto,
            fuel_mec,
            grounding_mec,
            fuel_pti,
            grounding_pti,
        ]
        return cost, mso, details

    def _obstacle_cost(self, location: geometry.Point) -> float:
        cost = 0
        for obstacle in self.obstacles.geoms:
            distance = self._obstacle_distance(location, obstacle)
            cost += np.exp(-self.zeta * distance)
        return cost

    def _path_deviation_cost(
        self, location: geometry.Point, waypoint_reference: geometry.Point
    ) -> float:
        return self._vector_weight(
            location.x - waypoint_reference.x,
            location.y - waypoint_reference.y,
        )

    def _segment_cost(
        self,
        location: geometry.Point,
        path_reference1: geometry.Point,
        path_reference2: geometry.Point,
    ) -> float:
        w1 = self._vector_weight(
            location.x - path_reference1.x, location.y - path_reference1.y
        )
        w2 = self._vector_weight(
            location.x - path_reference2.x, location.y - path_reference2.y
        )
        return (w1 - w2) ** 4

    def _wind_cost(self, point: geometry.Point) -> float:
        max_cost = self.wind_velocity * self.c_wind
        if point.intersects(self.obstacles):
            return max_cost
        cost = 0
        for obstacle in self.obstacles.geoms:
            closest, _ = ops.nearest_points(obstacle, point)
            vector = geometry.LineString((point, closest))
            distance = vector.length
            start, end = vector.coords
            dx, dy = [(end[i] - start[i]) / distance for i in (0, 1)]
            dot = dx * self.d_x + dy * self.d_y
            if dot > 0:
                distance_scaling = max_cost * np.exp(-self.zeta * distance)
                cost += dot * self.wind_velocity * distance_scaling
        return cost

    @staticmethod
    def _obstacle_distance(
        point: geometry.Point,
        obstacle: geometry.Polygon,
    ) -> float:
        if point.within(obstacle):
            return -point.distance(obstacle.exterior)
        return point.distance(obstacle)

    @staticmethod
    def _vector_weight(dx: float, dy: float) -> float:
        return dx**2 + dy**2
