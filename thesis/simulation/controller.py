"""
Contains the HeadingController for path following.
"""

import math

from shapely import geometry

from thesis import structures as st


class HeadingController:
    """
    Implements a heading-based path following controller for ship navigation.
    """

    horizon = 200

    @classmethod
    def follow(cls, path: st.Path, start: st.Waypoint) -> st.Trajectory:
        final_target = geometry.Point(path[-1].xy)
        location = geometry.Point(start.xy)
        velocity = 50
        poses = []
        path_line = geometry.LineString([wp.xy for wp in path])
        while location.distance(final_target) > cls.horizon:
            target = cls.get_next_target(location, path_line, cls.horizon)
            heading = cls.get_heading_to(target, location)
            poses.append(st.ShipPose(location.x, location.y, heading))
            location = cls.move_forward(velocity, location, heading)
        return st.Trajectory(poses)

    @staticmethod
    def get_next_target(
        location: geometry.Point, path: geometry.LineString, horizon: int
    ):
        intersection = path.intersection(location.buffer(horizon).exterior)
        if isinstance(intersection, geometry.Point):
            return intersection
        elif isinstance(intersection, geometry.MultiPoint):
            p1, p2 = intersection.geoms
            return p1 if p1.x < p2.x else p2

    @staticmethod
    def get_heading_to(
        target: geometry.Point, reference: geometry.Point
    ) -> float:
        return math.atan2(target.x - reference.x, target.y - reference.y)

    @staticmethod
    def move_forward(velocity: int, location: geometry.Point, heading: float):
        new_x = int(location.x + velocity * math.sin(heading))
        new_y = int(location.y + velocity * math.cos(heading))
        return geometry.Point(new_x, new_y)
