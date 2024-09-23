"""
Contains the Swarm class for Particle Swarm Optimization (PSO).
"""

from copy import copy

import numpy as np
from shapely import geometry

from thesis import models, structures as st
from thesis.pso import particle


class Swarm:
    """
    Implements the PSO procedure based on global best fitness and exploration.
    """

    iterations = 100
    segments = 20

    def __init__(self, index: int, setup: dict):
        self.waypoint_index = index
        self.disturbances = setup['disturbances']
        self.obstacles = setup['obstacles']
        self.references = self._init_point_references(setup['path'])
        self.guesses = self._init_route_guesses(setup['size'])
        self.yaw_angle = self._init_yaw_angle()
        self.particles = self._init_particles(setup['version'], setup['size'])
        self.optimal = False
        self.global_best_fitness = float('inf')
        self.global_best_position = None
        self.global_best_mso = 'pto'
        self.fitness_history = []
        self.position_history = []
        self.wkt = ()
        self.mso_costs = []

    @property
    def global_best(self) -> st.Waypoint:
        return st.Waypoint(
            self.global_best_position.x,
            self.global_best_position.y,
            self.global_best_mso,
        )

    def optimize_to_solution(self) -> None:
        for i in range(self.iterations):
            if self.optimal:
                print(
                    f'Waypoint {self.waypoint_index:02} '
                    f'| Optimality achieved: {self.wkt}'
                )
                return
            self._update_global_best()
            self._update_particles()
        print(
            f'Waypoint {self.waypoint_index:02} '
            f'| Max iteration limit: {self.wkt}'
        )

    def _init_particles(
        self, version: int, size: int
    ) -> list[particle.Particle]:
        config = dict(
            disturbances=self.disturbances,
            guess=self.references[0].coords[0],
            model=models.grounding_risks(self.obstacles, self.disturbances),
            obstacles=self.obstacles,
            ship_speed=2,
            yaw=self.yaw_angle,
        )
        particles = []
        for i in range(size):
            config['guess'] = self.guesses[i]
            particles.append(particle.Particle(version, config))
        return particles

    def _init_route_guesses(self, particles: int) -> list[tuple[float, float]]:
        ref0, ref1 = self.references[0], self.references[1]
        dx, dy = ref1.x - ref0.x, ref1.y - ref0.y
        guesses = [ref0.coords[0], ref0.coords[0]]
        for i in range(1, particles // 2 + 1):
            percentage = i / (particles // 2)
            left = ref0.x + dy * percentage, ref0.y - dx * percentage
            right = ref0.x - dy * percentage, ref0.y + dx * percentage
            guesses.append(left)
            guesses.append(right)
        return guesses

    def _init_yaw_angle(self) -> float:
        dx_0 = self.references[2].x - self.references[1].x
        dy_0 = self.references[2].y - self.references[1].y
        return np.arctan2(dx_0, dy_0)

    def _update_global_best(self) -> None:
        for p in self.particles:
            p.evaluate(*self.references)
            if p.fitness < self.global_best_fitness:
                self.global_best_fitness = float(p.fitness)
                self.global_best_position = copy(p.position)
                self.global_best_mso = p.mso
                self.wkt = p.wkt
                self.mso_costs = p.mso_costs
                if p.fitness < 1:
                    self.optimal = True
            self.fitness_history.append(self.global_best_fitness)
            self.position_history.append(self.global_best_position)

    def _update_particles(self) -> None:
        for p in self.particles:
            p.update_velocity(self.global_best_position)
            p.update_position()

    def _init_point_references(
        self, path: geometry.LineString
    ) -> tuple[geometry.Point, geometry.Point, geometry.Point]:
        percentage1 = (self.waypoint_index - 1) / self.segments
        percentage2 = (self.waypoint_index + 1) / self.segments
        wp_ref1 = path.interpolate(percentage1, normalized=True)
        wp_ref2 = path.interpolate(percentage2, normalized=True)
        refs_line = geometry.LineString((wp_ref1, wp_ref2))
        wp_ref0 = refs_line.interpolate(0.5, normalized=True)
        return wp_ref0, wp_ref1, wp_ref2
