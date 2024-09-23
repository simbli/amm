"""
Contains the Particle class for Particle Swarm Optimization (PSO).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from shapely import geometry

from thesis.pso import costs


class Particle:
    """
    Implements particle exploration and cost optimization behaviors for PSO.
    """

    INERTIA_WEIGHT = 0.75
    COGNITIVE_WEIGHT = 1
    SOCIAL_WEIGHT = 2
    V_LIMIT = 1

    def __init__(self, version: int, config: dict):
        self.version = version
        self.model = config['model']
        self.yaw = config['yaw']
        self.speed = config['ship_speed']
        self.bounds = _Bounds(*config['obstacles'].bounds)
        self.position = _Position(*config['guess'])
        self.velocity = _Velocity.random(self.V_LIMIT)
        self.mso = 'pto'
        self.best_mso = 'pto'
        self.best_position = self.position
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        self.wkt = ()
        self.mso_costs = []
        self.costs = costs.CostFunction(
            version, config['obstacles'], config['disturbances'][:2]
        )

    def estimate_grounding_probabilities(self) -> dict:
        x, y = self.position.x, self.position.y
        self.model.set_initial_states(y, x, self.yaw, self.speed)
        self.model.calculate_risk_output()
        return self.model.probabilities

    def evaluate(self, *references: geometry.Point) -> None:
        if self.version >= 16:
            probs = self.estimate_grounding_probabilities()
        else:
            probs = None
        results = self.costs.objectives(probs, self.position.xy, *references)
        self.fitness = sum(results['costs'])
        self.mso = results.get('mso', None)
        self.mso_costs = results.get('mso_costs', None)
        if self.fitness < self.best_fitness:
            self.best_position = self.position
            self.best_fitness = self.fitness
            self.best_mso = self.mso
            self.wkt = tuple(map(int, results['costs']))

    def update_velocity(self, global_best: _Position) -> None:
        v_x_cog = self._cognitive_velocity(self._delta_x(self.best_position))
        v_x_soc = self._social_velocity(self._delta_x(global_best))
        x_velocity = self.INERTIA_WEIGHT * self.velocity.x + v_x_cog + v_x_soc
        v_y_cog = self._cognitive_velocity(self._delta_y(self.best_position))
        v_y_soc = self._social_velocity(self._delta_y(global_best))
        y_velocity = self.INERTIA_WEIGHT * self.velocity.y + v_y_cog + v_y_soc
        self.velocity.set(x_velocity, y_velocity)

    def update_position(self) -> None:
        self.position.update(self.velocity, self.bounds)

    def _delta_x(self, best: _Position) -> float:
        return best.x - self.position.x

    def _delta_y(self, best: _Position) -> float:
        return best.y - self.position.y

    def _cognitive_velocity(self, delta: float) -> float:
        return self.COGNITIVE_WEIGHT * random.random() * self.V_LIMIT * delta

    def _social_velocity(self, delta: float) -> float:
        return self.SOCIAL_WEIGHT * random.random() * self.V_LIMIT * delta


@dataclass
class _Bounds:
    """
    Container data class for Particle bounds.
    """

    x_lower: float
    y_lower: float
    x_upper: float
    y_upper: float


@dataclass
class _Vector:
    """
    Container data class for two-dimensional spatial coordinates.
    """

    x: float
    y: float
    __slots__ = 'x', 'y'

    @property
    def xy(self) -> (int, int):
        return self.x, self.y


@dataclass
class _Position(_Vector):
    """
    Container data class for the two-dimensional position of a Particle.
    """

    @staticmethod
    def random(bounds: _Bounds):
        x = random.uniform(bounds.x_lower, bounds.x_upper)
        y = random.uniform(bounds.y_lower, bounds.y_upper)
        return _Position(x, y)

    def update(self, velocity: _Velocity, bounds: _Bounds):
        self.x = min(max(self.x + velocity.x, bounds.x_lower), bounds.x_upper)
        self.y = min(max(self.y + velocity.y, bounds.y_lower), bounds.y_upper)


@dataclass
class _Velocity(_Vector):
    """
    Container data class for the two-dimensional velocity of a Particle.
    """

    @staticmethod
    def random(limit: float):
        x, y = [random.uniform(-limit, limit) for _ in range(2)]
        return _Velocity(x, y)

    def set(self, x_velocity: float, y_velocity: float) -> None:
        self.x = x_velocity
        self.y = y_velocity
