"""
Contains the ANS class for simulating autonomous navigation of ships.
"""

import numpy as np
import seacharts

from thesis import models, simulation, structures as st
from thesis.ans import planner


class ANS:
    """
    Autonomous Navigation System - Plans and simulates the navigation process.
    """

    def __init__(self, article: str, figure: int):
        self.enc = self._enc_config_path(figure)
        self.enc.update()
        self.planner = planner.Planner(self.enc)
        self.simulator = simulation.Simulator(article, figure)

    def follow_path(self, number: int) -> st.Trajectory:
        path, *_ = self.planner.load_path_waypoints(number)
        return self.simulator.controller.follow(path, self.planner.start)

    def simulate_blackouts(self, trajectory: st.Trajectory) -> st.Trajectories:
        model = models.grounding_risks(
            self.enc.shore.geometry, self.simulator.disturbances
        )
        trajectories = []
        start_speed = 2
        for pose in trajectory:
            yaw_angle = pose.heading * np.pi / 180
            model.set_initial_states(pose.y, pose.x, yaw_angle, start_speed)
            model.calculate_risk_output()
            results = model.ttg_simulator.ship_model.simulation_results
            y_list = results['north position [m]']
            x_list = results['east position [m]']
            yaw_list = results['yaw angle [deg]']
            blackout_poses = list(zip(x_list, y_list, yaw_list))
            t = [st.ShipPose(p[0], p[1], p[2]) for p in blackout_poses]
            trajectories.append(st.Trajectory(t))
        return trajectories

    @staticmethod
    def trim(trajectories: st.Trajectories, start: int = 0) -> st.Trajectories:
        length = min([len(t) for t in trajectories])
        trajectories = [t[start:length:4] for t in trajectories]
        return trajectories

    @staticmethod
    def _enc_config_path(number: int) -> seacharts.ENC:
        if number < 14:
            enc_config = 1
        elif number == 18:
            enc_config = 3
        else:
            enc_config = 2
        path = st.files.enc_config_path(f'2023amm{enc_config}')
        return seacharts.ENC(path)
