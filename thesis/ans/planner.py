"""
Contains the Planner class for autonomous ship navigation planning.
"""

import numpy as np
import seacharts
import shapely
from shapely import geometry

from thesis import structures as st


class Planner:
    """
    Plans paths and waypoints for ship routes using SeaCharts for obstacles.
    """

    def __init__(self, enc: seacharts.ENC):
        self.enc = enc
        self.obstacles = self._init_obstacles()
        self.waypoints = self._init_route_waypoints()
        self.start, self.target = self.waypoints[1:3]
        self.paths, self.route_obstructions = self._plan_paths()

    def plan_ideal_route_trajectory(self, index: int) -> st.Trajectory:
        ideal_poses = []
        segments = 20
        step = 1 / segments
        path = self.paths[index]
        waypoints = [
            path.interpolate(step * i, normalized=True)
            for i in range(segments)
        ]
        points = [*[wp.coords[0] for wp in waypoints], self.target.xy]
        dx_0 = points[1][0] - points[0][0]
        dy_0 = points[1][1] - points[0][1]
        headings = [np.arctan2(dx_0, dy_0)]
        for wp_past, wp_next in zip(points[:-2], points[2:]):
            dx = wp_next[0] - wp_past[0]
            dy = wp_next[1] - wp_past[1]
            angle = np.arctan2(dx, dy)
            headings.append(angle * 180 / np.pi)
        for wp, heading in zip(waypoints, headings):
            x, y = wp.coords[0]
            ideal_poses.append(st.ShipPose(x, y, heading))
        return st.Trajectory(ideal_poses[1:13])

    def plan_route_alternatives(self, *points: tuple[float, float]) -> dict:
        start, target = [st.Waypoint(*p) for p in points]
        alternatives = {}
        bbox = self.enc.bbox
        seabed = self.enc.seabed[10].geometry
        horizon = geometry.LineString(zip(bbox[0::2], bbox[1::2])).envelope
        obstacles = horizon.difference(seabed).buffer(10).simplify(10)
        path_line = geometry.LineString([start.xy, target.xy])
        intersects = [o for o in obstacles.geoms if o.intersects(path_line)]
        line = geometry.LineString([start.xy, target.xy])
        buffered_convex = [o.convex_hull for o in obstacles.buffer(50).geoms]
        intersecting = [o for o in buffered_convex if line.intersects(o)]
        obstacle = intersecting[0].simplify(10)
        lefts, rights = self._visible_locations(obstacle, line)
        left, right = self._distance_arrows(lefts, rights, line)
        sides = self._plan_alternative_paths(start, target, obstacle)
        alternatives['sides'] = sides
        alternatives['start'] = start
        alternatives['target'] = target
        alternatives['intersects'] = intersects
        alternatives['obstacle'] = obstacle
        alternatives['left_arrow'] = left
        alternatives['left_points'] = lefts
        alternatives['right_arrow'] = right
        alternatives['right_points'] = rights
        return alternatives

    @staticmethod
    def load_path_waypoints(*variants: int) -> st.Paths:
        tuples = [st.files.load_path_waypoints(i) for i in variants]
        return st.Paths([[st.Waypoint(*p) for p in wp] for wp in tuples])

    def _distance_arrows(
        self, lefts: st.Path, rights: st.Path, line: geometry.LineString
    ) -> list[st.Path]:
        sides = []
        for side in (lefts, rights):
            end = self._max_distance(side, line)
            start = line.interpolate(line.project(geometry.Point(end.xy)))
            arrow = geometry.LineString([start.coords[0], end.xy])
            sides.append(
                [
                    st.Waypoint(*arrow.interpolate(v).coords[0])
                    for v in (10, arrow.length - 35)
                ]
            )
        return sides

    def _init_obstacles(self) -> geometry.MultiPolygon:
        horizon = shapely.LineString(
            zip(self.enc.bbox[0::2], self.enc.bbox[1::2])
        ).envelope
        geom = horizon.difference(self.enc.seabed[5].geometry)
        margin = 10
        obstacles = geom.buffer(margin).simplify(margin)
        if isinstance(obstacles, geometry.Polygon):
            shapes = [geom.convex_hull]
        else:
            shapes = [g.convex_hull for g in obstacles.geoms]
        obstacles = geometry.MultiPolygon(shapes)
        return obstacles

    def _path_obstacle(self, *wp: st.Waypoint) -> geometry.Polygon:
        buffer_radius = 100
        line = geometry.LineString([p.xy for p in wp])
        layer = self.enc.seabed[10].geometry
        horizon = layer.envelope
        obstacles = horizon.difference(layer)
        buffered_convex = [
            o.convex_hull for o in obstacles.buffer(buffer_radius).geoms
        ]
        intersecting = [o for o in buffered_convex if line.intersects(o)]
        if not intersecting:
            return geometry.Polygon()
        obstacle = intersecting[0].simplify(10)
        return obstacle

    def _plan_paths(self) -> tuple[list, list]:
        obstacle1 = self._path_obstacle(self.start, self.target)
        if not obstacle1:
            return [], []
        setup1 = self.start, self.target, obstacle1
        paths = self._plan_alternative_paths(*setup1)
        start2, target2 = paths[0][1], paths[0][2]
        obstacle2 = self._path_obstacle(start2, target2)
        setup2 = start2, target2, obstacle2
        paths2 = self._plan_alternative_paths(*setup2)
        paths[0] = paths[0][:1] + paths2[1][:-1] + paths[0][2:]
        path_points = [[wp.xy for wp in path] for path in paths]
        path_lines = [geometry.LineString(points) for points in path_points]
        return path_lines, [obstacle1, obstacle2]

    def _plan_alternative_paths(
        self,
        start: st.Waypoint,
        target: st.Waypoint,
        obstacle: geometry.Polygon,
    ) -> st.Paths:
        sides = [[start], [start]]
        for i, side in enumerate(sides):
            while side[-1] != target:
                last_point = side[-1]
                if self._is_visible_from(target.xy, last_point.xy, obstacle):
                    side.append(target)
                    continue
                new_line = geometry.LineString([last_point.xy, target.xy])
                candidates = self._visible_locations(obstacle, new_line)
                next_waypoint = self._max_distance(candidates[i], new_line)
                side.append(next_waypoint)
        return st.Paths(sides)

    def _visible_locations(
        self,
        polygon: geometry.Polygon,
        line: geometry.LineString,
    ) -> tuple[list, list]:
        left, right = [], []
        obstacle = polygon.buffer(1)
        start, end = line.coords[0], line.coords[1]
        for point in obstacle.exterior.coords:
            if not self._is_visible_from(point, line.coords[0], polygon):
                continue
            rotation = geometry.LinearRing([start, point, end])
            side = right if rotation.is_ccw else left
            distances_to_existing_locations = [
                geometry.Point(point).distance(geometry.Point(p.xy))
                for p in side
            ]
            if all([d > 10 for d in distances_to_existing_locations]):
                side.append(st.Waypoint(*point))
        return left, right

    @staticmethod
    def _init_route_waypoints() -> st.Path:
        tuples = st.files.read_route_waypoints(1)
        return st.Path(st.Waypoint(*p) for p in tuples)

    @staticmethod
    def _is_visible_from(
        target: tuple[float, float],
        reference: tuple[float, float],
        polygon: geometry.Polygon,
    ) -> bool:
        sight_line = geometry.LineString([reference, target])
        return not sight_line.intersects(polygon)

    @staticmethod
    def _max_distance(
        points: st.Path, line: geometry.LineString
    ) -> st.Waypoint:
        return max(points, key=lambda p: [geometry.Point(p.xy).distance(line)])
