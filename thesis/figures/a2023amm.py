"""
Contains functions for creating all figures of the AMM journal article.
"""

import time

from thesis import ans, pso, structures as st
from thesis.figures import plots


def fig8(_) -> None:
    """Study area with preplanned ship route and obstacle collision area."""
    _setup(8)


def fig9(_) -> None:
    """End result of the Level-1 route planning algorithm of Appendix A."""
    _ans, plot = _setup(9)
    plot.red_disks(_ans.planner.start, _ans.planner.target)
    plot.highlight(_ans.planner.route_obstructions)
    plot.planned_paths(_ans.planner.paths)


def fig10(run: bool) -> None:
    """PSO demonstration of waypoint paths optimized along the ship route."""
    _ans, plot = _setup(10)
    plot.red_disks(_ans.planner.start, _ans.planner.target)
    plot.previous_paths(_ans.planner.paths)
    if run:
        _run_optimization(_ans, version=10, size=50, k=1)
    paths = _ans.planner.load_path_waypoints(1, 2)
    plot.locations(paths[0], 'yellow')
    plot.locations(paths[1], 'pink')


def fig11(_) -> None:
    """Colored contours of the distance-based grounding risk cost term."""
    _ans, plot = _setup(11)
    color_variants = 'abcdefg'
    previous = _ans.planner.obstacles
    try:
        # Plots custom colors if added to the SeaCharts color palette
        plot.draw(previous.geoms, color_variants[-1])
        for i in range(1, len(color_variants) - 1):
            color = color_variants[-i - 1]
            current = previous.buffer(100)
            layer = current.difference(previous).geoms
            plot.draw(layer, color)
            previous = current
    except ValueError:
        plot.custom_colors_message(color_variants)
    paths = _ans.planner.load_path_waypoints(1, 2)
    plot.locations(paths[0], 'yellow')
    plot.locations(paths[1], 'cyan')


def fig12(run: bool) -> None:
    """Comparison with the added scalar product wind disturbance cost term."""
    _ans, plot = _setup(12)
    plot.disturbances(_ans.enc.bbox, _ans.simulator.disturbances)
    plot.previous_paths(_ans.planner.paths)
    paths = _ans.planner.load_path_waypoints(3, 4)
    plot.locations(paths[0], 'orange')
    plot.locations(paths[1], 'magenta')
    if run:
        _run_optimization(_ans, version=12, size=50, k=5)
    paths = _ans.planner.load_path_waypoints(5, 6)
    plot.locations(paths[0], 'yellow')
    plot.locations(paths[1], 'pink')
    plot.number(*paths)


def fig13(_) -> None:
    """Path following using a simple line-of-sight guidance controller."""
    _ans, plot = _setup(13)
    trajectories = [_ans.follow_path(1), _ans.follow_path(2)]
    trajectories = _ans.trim(trajectories, start=8)
    plot.trajectory(trajectories[0], 'yellow')
    plot.trajectory(trajectories[1], 'pink')


def fig14(_) -> None:
    """Time-to-grounding predictions for LOPP scenarios along each path."""
    _ans, plot = _setup(14)
    plot.disturbances(_ans.enc.bbox, _ans.simulator.disturbances)
    path_colors = [('orange', 'yellow'), ('green', 'pink')]
    for i, colors in enumerate(path_colors):
        trajectory = _ans.planner.plan_ideal_route_trajectory(i)
        blackouts = _ans.simulate_blackouts(trajectory)
        plot.blackout_traces(trajectory, blackouts, colors)


def fig15(_) -> None:
    """Virtual risk gradient arrows for the time-to-grounding cost term."""
    _ans, plot = _setup(15)
    plot.disturbances(_ans.enc.bbox, _ans.simulator.disturbances)
    path_colors = [('yellow', 'orange'), ('pink', 'green')]
    for i, colors in enumerate(path_colors):
        trajectory = _ans.planner.plan_ideal_route_trajectory(i)
        blackouts = _ans.simulate_blackouts(trajectory)
        wind = _ans.simulator.disturbances[:2]
        plot.blackout_gradients(trajectory, blackouts, wind, colors)


def fig16(run: bool) -> tuple[list, plots.Plot]:
    """Resulting path and trajectories using the complete cost function."""
    _ans, plot = _setup(16)
    plot.disturbances(_ans.enc.bbox, _ans.simulator.disturbances)
    plot.previous_paths(_ans.planner.paths)
    if run:
        costs = _run_optimization(_ans, version=16, size=50, k=7)
        st.files.save_path_costs(costs)
    else:
        costs = st.files.load_path_costs()
    paths = _ans.planner.load_path_waypoints(7, 8)
    plot.number(*paths)
    waypoints = {}
    for path, path_color in zip(paths, ['yellow', 'pink']):
        for i, waypoint in enumerate(path):
            match waypoint.mso:
                case 'mec':
                    color = 'cyan'
                case 'pti':
                    color = 'purple'
                case _:
                    color = path_color
            if color not in waypoints:
                waypoints[color] = []
            waypoints[color].append(waypoint)
    for color, wp in waypoints.items():
        plot.locations(wp, color)
    trajectories = [_ans.follow_path(7), _ans.follow_path(8)]
    trajectories = _ans.trim(trajectories, start=4)
    mec_poses = []
    # Concept visualization for a proper MSO mode selection algorithm
    for i, trajectory in enumerate(trajectories):
        for j, pose in enumerate(trajectory[:], start=1):
            if i == 0 and 6 <= j <= 7 or i == 1 and 4 <= j <= 4:
                mec_poses.append(pose)
                trajectory.remove(pose)
    plot.trajectory(trajectories[0], 'orange')
    plot.trajectory(trajectories[1], 'green')
    plot.trajectory(mec_poses, 'red')
    return costs, plot


def fig17(run: bool) -> None:
    """Weighted MSO modes and accumulated route costs for all waypoints."""
    costs, plot = fig16(run)
    columns = ['Fuel cost', 'Grounding cost']
    frames, cumulative = [], []
    for wp_costs in costs:
        indexes = list(range(1, len(wp_costs) + 1))
        totals = [(v[0] + v[1], v[2] + v[3], v[4] + v[5]) for v in wp_costs]
        total = []
        total_sum = 0
        for values in totals:
            total_sum += min(values)
            total.append(total_sum)
        cumulative.append(total)
        maximum = max([max(v) for v in totals])
        normalized = [[v / maximum for v in values] for values in wp_costs]
        frames.append((normalized, indexes, columns))
    yellow, pink = [c[-1] for c in cumulative]
    color = 'yellow' if yellow < pink else 'pink'
    print(f'The estimated most cost efficient route is the {color} option.')
    plot.stacked_clusters(frames, cumulative)


def fig18(_) -> None:
    """Alternative routes visualization of path planning Algorithm A1."""
    _ans, plot = _setup(18)
    start = 153700, 7017200
    target = 155700, 7019200
    # The simplified path planning algorithm of Figure A1
    paths = _ans.planner.plan_route_alternatives(start, target)
    plot.highlight(paths['obstacle'])
    plot.obstructions(paths['intersects'])
    plot.locations(paths['left_points'][1:-1], 'magenta', 30)
    plot.locations(paths['right_points'][:1], 'orange', 30)
    plot.line(paths['start'], paths['target'], color='green')
    for side_points, color in zip(paths['sides'], ['pink', 'yellow']):
        plot.line(*side_points, color=color)
        plot.locations(side_points[1:-1], color, radius=30)
    plot.path_arrow(*paths['left_arrow'], color='cyan')
    plot.path_arrow(*paths['right_arrow'], color='cyan')
    plot.circle(*target, radius=40, color='green')
    plot.ship(*start, heading=0, color='white')


def _setup(number: int) -> tuple[ans.ANS, plots.Plot]:
    """Sets up the ANS and Plot classes for all figure creation functions."""
    _ans = ans.ANS('2023amm', number)
    plot = plots.Plot(
        _ans.enc.display,
        _ans.planner.obstacles,
        _ans.enc.shore.geometry,
        _ans.planner.waypoints,
    )
    plot.route(
        links=number < 11,
        collision=number == 8,
        main_path=number < 14,
    )
    return _ans, plot


def _run_optimization(_ans: ans.ANS, version: int, size: int, k: int) -> list:
    """Sets up and runs the PSO algorithm for generating ship trajectories."""
    algorithm = pso.PSO(
        _ans.enc, _ans.simulator.disturbances, _ans.planner.obstacles
    )
    start = time.time()
    paths = algorithm.run(version, size, *_ans.planner.paths)
    print(f'Finished optimization in {int(time.time() - start)} seconds.')
    for i, path in enumerate(paths, start=k):
        waypoints = [(wp.x, wp.y, wp.mso) for wp in path]
        st.files.save_path_waypoints(waypoints, i)
    return algorithm.costs
