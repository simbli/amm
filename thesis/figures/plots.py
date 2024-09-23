"""
Contains the Plot class for plotting article figures.
"""

from collections.abc import Callable

import pandas as pd
from matplotlib import pyplot as plt, colors as mc, cm
import numpy as np
import seacharts
import shapely
from shapely import affinity, geometry

from thesis import structures as st


class Plot:
    """
    Plots geometry and graphs using the SeaCharts Display as its canvas.
    """

    def __init__(
        self,
        display: seacharts.enc.Display,
        obstacles: geometry.MultiPolygon,
        shoreline: geometry.MultiPolygon,
        waypoints: list[st.Waypoint],
    ):
        self.display = display
        self.display.dark_mode()
        self.obstacles = obstacles
        self.shoreline = shoreline
        self.waypoints = waypoints

    def blackout_gradients(
        self,
        trajectory: st.Trajectory,
        blackouts: list[st.Trajectory],
        disturbances: tuple[float, float],
        colors: tuple[str, str],
    ) -> None:
        ships = self._ship_colors(trajectory, colors[0])
        lines, red_pairs = [], []
        for start, poses in zip(trajectory, blackouts):
            new = self._ship_colors(poses, colors[1])
            reds = [s for s in new if s[1] == 'red']
            if reds:
                grounding = reds[0]
                index = grounding[2]
                red_pairs.append((start.xy, poses[index].xy))
                ships.append(grounding)
                lines.append(self._blackout_trace(start, poses[: len(new)]))
        self._trace(ships, lines)
        for start, end in red_pairs:
            self._arrows(start, end, colors[1], disturbances)

    def blackout_traces(
        self,
        trajectory: st.Trajectory,
        blackouts: list[st.Trajectory],
        colors: tuple[str, str],
    ) -> None:
        ships = self._ship_colors(trajectory, colors[0])
        lines = []
        for start, poses in zip(trajectory, blackouts):
            new = self._ship_colors(poses, colors[1], only_even=True)
            end = len(new) * 2 + 1
            lines.append(self._blackout_trace(start, poses[:end], step=2))
            ships += new
        self._trace(ships, lines)

    def circle(self, x: float, y: float, radius: int, color: str) -> None:
        self.display.draw_circle((x, y), radius, color)

    def disturbances(self, bbox: tuple, disturbances: tuple) -> None:
        c_x = bbox[2] - 400
        c_y = bbox[3] - 400
        center = c_x, c_y
        wind_angle_deg, wind_velocity, currents_velocity = disturbances
        angle_rad = wind_angle_deg * np.pi / 180
        unit = np.sin(angle_rad), np.cos(angle_rad)
        radius, color = 200, 'white'
        arrow_tip = (
            c_x + (radius - 15) * unit[0],
            c_y + (radius - 15) * unit[1],
        )
        self.display.draw_circle(center, radius, 'full_horizon')
        self.display.draw_circle(
            center, radius, 'black', fill=False, thickness=3.0
        )
        self.display.draw_arrow(
            center, arrow_tip, color, width=20, head_size=80
        )
        letters = ['E', 'N', 'W', 'S']
        orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for letter, (dx, dy) in zip(letters, orientations):
            x, y = c_x + dx * radius * 1.3, c_y + dy * radius * 1.3
            self._add_white_text(letter, x, y - 12)
        string = str(wind_velocity) + ' m/s'
        self._add_white_text(string, c_x, c_y + radius / 2)
        if currents_velocity:
            string = str(currents_velocity) + ' m/s'
            self._add_white_text(string, c_x, c_y - radius / 2 - 40)

    def draw(self, shape: geometry.Polygon, color: str) -> None:
        self.display.draw_polygon(shape, color)

    def highlight(self, *shapes: geometry.Polygon) -> None:
        for shape in shapes:
            self.display.draw_polygon(shape, 'full_horizon')

    def line(self, *points: st.Waypoint, color: str) -> None:
        self.display.draw_line([p.xy for p in points], color=color)

    def locations(
        self,
        path: st.Path,
        color: str,
        radius: int = 20,
    ) -> None:
        for location in path:
            self.display.draw_circle(location.xy, radius, color)

    def number(self, *paths: st.Path) -> None:
        for path in paths:
            for j, location in enumerate(path, start=1):
                x, y = [v + 20 for v in location.xy]
                self.display.axes.text(x, y, j, size=15, color='white')

    def obstructions(self, obstacles: list[geometry.Polygon]) -> None:
        for o in obstacles:
            self.display.draw_polygon(o.convex_hull.buffer(50), 'red')

    def path_arrow(self, *points: st.Waypoint, color: str) -> None:
        self.display.draw_arrow(
            points[0].xy, points[1].xy, color, width=10, head_size=80
        )

    def planned_paths(self, paths: tuple[geometry.LineString]) -> None:
        for path, color in zip(paths, ['yellow', 'pink']):
            side_points = path.coords
            self.display.draw_line(side_points, color)
            for point in side_points[1:-1]:
                self.display.draw_circle(point, 30, color)

    def previous_paths(self, paths: tuple[geometry.LineString]) -> None:
        for path in paths:
            line = path.buffer(10)
            points = geometry.MultiPoint(path.coords[1:-1]).buffer(30)
            self.display.draw_polygon(line.union(points), 'lightgrey')

    def red_disks(self, *points: st.Waypoint) -> None:
        for point in points:
            self.display.draw_circle(point.xy, 30, 'red')

    def route(self, links: bool, collision: bool, main_path: bool) -> None:
        lines = []
        for wp1, wp2 in zip(self.waypoints, self.waypoints[1:]):
            lines.append(geometry.LineString([wp1.xy, wp2.xy]).buffer(40))
        if links:
            points = [geometry.Point(wp.xy) for wp in self.waypoints]
            disks = [p.buffer(120) for p in points]
            holes = [p.buffer(40) for p in points]
            path = shapely.unary_union(lines + disks)
            for i, hole in enumerate(holes):
                path = path.difference(hole)
        else:
            lines.pop(1)
            path = shapely.unary_union(lines)
        if collision:
            overlap = path.intersection(self.obstacles)
            self.display.draw_polygon(overlap, 'red')
            path = path.difference(self.obstacles)
        if main_path:
            self.display.draw_polygon(path, 'green')

    def ship(self, x: float, y: float, heading: float, color: str) -> None:
        ship = _ShipDrawing(x, y, heading).shape
        self.display.draw_polygon(ship, color)

    def trajectory(self, poses: st.Trajectory, color: str) -> None:
        for pose in poses:
            x, y, heading = pose.as_tuple
            self.ship(x, y, heading * 180 / np.pi, color)

    def _add_white_text(self, string: str, x: float, y: float) -> None:
        self.display.axes.text(
            x, y, string, size=20, ha='center', va='center', color='w'
        )

    def _arrows(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        disturbances: tuple[float, float],
    ) -> None:
        wind_function = self._wind_setup(*disturbances)
        vector = geometry.LineString([start, end])
        if vector.length > 100:
            size = -300 / vector.length
            scaling = wind_function(size, vector)
            if scaling < -0.1:
                arrow = affinity.scale(vector, scaling, scaling, origin=start)
                p1, p2 = arrow.coords[:2]
                self.display.draw_arrow(p1, p2, color)

    def _ship_colors(
        self,
        poses: st.Trajectory,
        color: str,
        only_even: bool = False,
    ) -> list[tuple[geometry.Polygon, str, int]]:
        step = 2
        colored_ships = []
        for j, pose in enumerate(poses):
            x, y, yaw = pose.as_tuple
            ship = _ShipDrawing(x, y, yaw).shape
            if ship.buffer(3).intersects(self.shoreline):
                colored_ships.append((ship, 'red', j))
                break
            if not only_even or j >= step and j % step == 0:
                colored_ships.append((ship, color, j))
        return colored_ships

    def _trace(
        self,
        ships: list[tuple[geometry.Polygon, str, int]],
        lines: list[geometry.Polygon],
    ) -> None:
        for polygon, color, _ in ships:
            self.draw(polygon, color)
        traces = geometry.MultiPolygon(lines).buffer(0)
        collection = geometry.MultiPolygon([s[0] for s in ships])
        visual = traces.difference(collection.buffer(0)).geoms
        self.highlight(visual)

    @staticmethod
    def custom_colors_message(color_variants: str) -> None:
        jet = plt.get_cmap('YlOrRd')
        c_norm = mc.Normalize(vmin=0, vmax=len(color_variants))
        scalar_map = cm.ScalarMappable(norm=c_norm, cmap=jet)
        print('_custom_colors = dict(')
        for i in range(len(color_variants)):
            color = scalar_map.to_rgba(np.array(i))
            s = ', '.join([str(round(v, 3)) for v in color[:3]])
            print(f'    {color_variants[i]} = (({s}, 0.8), ({s}, 0.4)),')
        print(')\n')
        print(
            f'Add the above printed dictionary to `seacharts.display.colors`'
            f' in order to see the gradient colors shown in the article.'
        )

    @staticmethod
    def stacked_clusters(frames: list[tuple], cumulative: list[list]) -> None:
        panda_frames = [
            [
                pd.DataFrame(
                    np.array([wp[2 * i: 2 * (i + 1)] for wp in frame[0]]),
                    index=frame[1],
                    columns=frame[2],
                )
                for i in range(3)
            ]
            for frame in frames
        ]
        n_df = len(panda_frames[0])
        n_col = len(panda_frames[0][0].columns)
        n_ind = len(panda_frames[0][0].index)
        fig, axs = plt.subplots(2, figsize=(8, 6))
        color_names = ['cornflowerblue', 'turquoise']
        color_map = mc.LinearSegmentedColormap.from_list('', color_names)
        _max = max([c[-1] for c in cumulative])
        for k, frames in enumerate(panda_frames):
            axe, df = None, None
            for df in frames:
                axe = df.plot(
                    kind='bar',
                    linewidth=0,
                    stacked=True,
                    ax=axs[k],
                    legend=False,
                    grid=False,
                    cmap=color_map,
                )
            h, legends = axe.get_legend_handles_labels()
            for i in range(0, n_df * n_col, n_col):
                for j, pa in enumerate(h[i: i + n_col]):
                    for rect in pa.patches:
                        rect.set_x(rect.get_x() + 1 / (n_df + 1) * i / n_col)
                        rect.set_hatch('/' * int(i / n_col))
                        rect.set_width(1 / float(n_df + 1))
            axe.set_xticks(
                (np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0
            )
            axe.set_xticklabels(df.index, rotation=0)
            color_names = ['gold', 'magenta']
            ax2 = axe.twinx()
            ax2.plot(
                list(range(len(cumulative[k]))),
                [v / _max for v in cumulative[k]],
                color=color_names[k],
            )
            axe.set_ylim([0.0, 1.1])
            ax2.set_ylim([0.0, 1.1])
            if k == 0:
                axe.set_title('MSO costs per waypoint')
        h, legends = axs[0].get_legend_handles_labels()
        l1 = axs[0].legend(h[:n_col], legends[:n_col], loc='upper left')
        axs[0].add_artist(l1)
        n = []
        for i in range(n_df):
            n.append(axs[1].bar(0, 0, color='lightgray', hatch='/' * i))
        plt.legend(n, ['PTO', 'MEC', 'PTI'], loc='upper left')
        axs[1].set_xlabel('Waypoint indexes')
        plt.figure(fig.number)

    @staticmethod
    def _blackout_trace(
        start: st.ShipPose,
        poses: st.Trajectory,
        step: int = 1,
    ) -> geometry.Polygon:
        first = [start.xy]
        last = [poses[-1].xy]
        points = first + [p.xy for p in poses[step:-1:step]] + last
        disks = geometry.MultiPolygon(
            [geometry.Point(p).buffer(8) for p in points]
        )
        line = geometry.LineString(points).buffer(3)
        polygon = line.union(disks.buffer(0))
        return polygon

    @staticmethod
    def _wind_setup(angle_deg: float, velocity: float) -> Callable:
        angle_rad = angle_deg * np.pi / 180
        disturbance = np.sin(angle_rad), np.cos(angle_rad)

        def wind_gradients(size, vector):
            d = vector.length
            start, end = vector.coords
            dx = (end[0] - start[0]) / d
            dy = (end[1] - start[1]) / d
            dot = dx * disturbance[0] + dy * disturbance[1]
            return size * dot * velocity * np.exp(-d / 400) / 6

        return wind_gradients


class _ShipDrawing:
    """
    Constructs the ship geometry for plot drawing using Shapely.
    """

    width = 15
    length = 83

    def __init__(self, x: float, y: float, heading_rad: float):
        x_min, x_max, y_min, y_max, top = (
            x - self.width / 2,
            x + self.width / 2,
            y - self.length / 2,
            y + self.length / 2 - self.width,
            (x, y + self.length / 2),
        )
        coordinates = (
            (x_min, y_min),
            (x_min, y_max),
            top,
            (x_max, y_max),
            (x_max, y_min),
        )
        self.shape = affinity.rotate(
            geometry.Polygon(coordinates),
            angle=-heading_rad,
            origin=(x, y),
        )
