"""
Contains utility functions for working with OS files and paths.
"""

import csv
import pathlib


top_level = pathlib.Path.cwd()
data_path = top_level / 'data'
output_path = top_level / 'output'

config_path = data_path / 'config'
solutions_path = data_path / 'solutions'

enc_path = config_path / 'enc'
routes_path = config_path / 'routes'

pso_path = solutions_path / 'pso'


def enc_config_path(article: str) -> pathlib.Path:
    return enc_path / f'config{article}.yaml'


def load_path_costs() -> list[list[tuple]]:
    values_pair = []
    for index in range(1, 3):
        with open(pso_path / f'costs{index}.csv', 'r') as file:
            rows = [r for r in csv.reader(file, delimiter=',')]
        costs = []
        for row in rows:
            costs.append(tuple(map(float, row)))
        values_pair.append(costs)
    return values_pair


def load_path_waypoints(number: int) -> list[tuple]:
    waypoints = []
    with open(pso_path / f'path{number}.csv', 'r') as file:
        rows = [r for r in csv.reader(file, delimiter=',')]
    for row in rows:
        x, y = tuple(map(int, row[:2]))
        mso = row[2] if len(row) > 2 else ''
        waypoints.append((x, y, mso, 0))
    return waypoints


def read_route_waypoints(number: int) -> list[tuple]:
    last_x, last_y, accumulative_length = 0, 0, 0

    def euclidean_distance(x_delta, y_delta):
        return (x_delta**2 + y_delta**2) ** (1 / 2)

    waypoints = []
    with open(routes_path / f'route{number}.csv') as file:
        rows = [r for r in csv.reader(file, delimiter=',')]
    for row in rows:
        values = [v.replace(' ', '') for v in row]
        x, y = map(int, values[:2])
        mso = values[2] if len(values) > 2 else ''
        accumulative_length += euclidean_distance(x - last_x, y - last_y)
        waypoints.append((x, y, mso, accumulative_length))
        last_x, last_y = x, y
    return waypoints


def save_path_costs(values_pair: list[list[tuple]]) -> None:
    for index, values in enumerate(values_pair, start=1):
        file_path = pso_path / f'costs{index}.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(values)


def save_path_waypoints(waypoints: list[tuple], number: int) -> None:
    file_path = pso_path / f'path{number}.csv'
    points = [(int(wp[0]), int(wp[1]), *wp[2:]) for wp in waypoints]
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points)
