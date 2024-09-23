"""
Contains the PSO class for Particle Swarm Optimization (PSO).
"""

import multiprocessing as mp
import random

import seacharts
from shapely import geometry

from thesis import structures as st
from thesis.pso import swarm

global multiprocessing_setup


class PSO:
    """
    Initializes and runs parallel PSO swarms using multiprocessing.
    """

    def __init__(
        self,
        enc: seacharts.ENC,
        disturbances: tuple[float, float, float],
        obstacles: geometry.MultiPolygon,
    ):
        self.enc = enc
        self.disturbances = disturbances
        self.obstacles = obstacles
        self.solutions = []
        self.costs = []

    def run(self, version: int, size: int, *paths: st.Path) -> st.Paths:
        for i, path in enumerate(paths):
            print(f'Running PSO version {version} for path {i + 1}...')
            solution = self._parallel_optimize(version, size, path)
            print(f'Path {i + 1} completed.\n')
            self.solutions.append(solution)
        return self.solutions

    def _parallel_optimize(
        self, version: int, particles: int, path: st.Path
    ) -> st.Path:
        args = (
            dict(
                version=version,
                path=path,
                obstacles=self.obstacles,
                disturbances=self.disturbances,
                size=particles,
            ),
        )
        with mp.Pool(initializer=self._init_workers, initargs=args) as pool:
            swarms = pool.map(self._optimize, range(1, swarm.Swarm.segments))
        self.costs.append([s.mso_costs for s in swarms])
        return st.Path(s.global_best for s in swarms)

    @staticmethod
    def _init_workers(setup: dict) -> None:
        global multiprocessing_setup
        multiprocessing_setup = setup

    @staticmethod
    def _optimize(i: int) -> swarm.Swarm:
        random.seed(0)
        particle_swarm = swarm.Swarm(i, multiprocessing_setup)
        particle_swarm.optimize_to_solution()
        return particle_swarm
