"""
Contains setup functions for initializing the risks.GroundingRiskModel.
"""

import numpy as np
from shapely import geometry

from thesis.models import risks, transit


def grounding_risks(
    obstacles: geometry.base.BaseGeometry,
    disturbances: tuple[float, float, float],
) -> risks.GroundingRiskModel:
    """Returns a risks.GroundingRiskModel with obstacles and disturbances."""
    direction_deg, wind_speed, current_speed = disturbances
    direction_rad = direction_deg * np.pi / 180
    currents_north = current_speed * np.cos(direction_rad)
    currents_east = current_speed * np.sin(direction_rad)
    config = dict(
        max_sim_time=600,
        time_interval=600,
        integration_step=30,
        wind_speed=wind_speed,
        wind_direction=direction_rad,
        currents_north=currents_north,
        currents_east=currents_east,
    )
    setup = _set_up_simulations(obstacles, config)
    risk_model = risks.GroundingRiskModel(**setup)
    return risk_model


def _set_up_simulations(
    obstacles: geometry.base.BaseGeometry, config: dict
) -> dict:
    """Returns the simulation settings for the risks.GroundingRiskModel."""
    setup = dict(
        ttg_sim_config=transit.ShipConfiguration(
            coefficient_of_deadweight_to_displacement=0.7,
            bunkers=200000,
            ballast=200000,
            length_of_ship=80,
            width_of_ship=16,
            added_mass_coefficient_in_surge=0.4,
            added_mass_coefficient_in_sway=0.4,
            added_mass_coefficient_in_yaw=0.4,
            dead_weight_tonnage=3850000,
            mass_over_linear_friction_coefficient_in_surge=130,
            mass_over_linear_friction_coefficient_in_sway=18,
            mass_over_linear_friction_coefficient_in_yaw=90,
            nonlinear_friction_coefficient__in_surge=2400,
            nonlinear_friction_coefficient__in_sway=4000,
            nonlinear_friction_coefficient__in_yaw=400,
        ),
        scenario_params=risks.ScenarioAnalysisParameters(
            main_engine_failure_rate=3e-9,
            genset_one_failure_rate=6e-9,
            genset_two_failure_rate=6e-9,
            hsg_failure_rate=2e-9,
            main_engine_mean_start_time=50,
            main_engine_start_time_std=1.2,
            main_engine_start_time_shift=20,
            main_engine_nominal_start_prob=1,
            main_engine_mean_restart_time=50,
            main_engine_restart_time_std=1.2,
            main_engine_restart_time_shift=20,
            main_engine_nominal_restart_prob=0.4,
            genset_one_mean_start_time=35,
            genset_one_start_time_std=1.0,
            genset_one_start_time_shift=14,
            genset_one_nominal_start_prob=1,
            genset_two_mean_start_time=35,
            genset_two_start_time_std=1.0,
            genset_two_start_time_shift=14,
            genset_two_nominal_start_prob=1,
            genset_one_mean_restart_time=35,
            genset_one_restart_time_std=1.0,
            genset_one_restart_time_shift=14,
            genset_one_nominal_restart_prob=0.5,
            genset_two_mean_restart_time=35,
            genset_two_restart_time_std=1.0,
            genset_two_restart_time_shift=14,
            genset_two_nominal_restart_prob=0.5,
            hsg_mean_start_time=12,
            hsg_start_time_std=1,
            hsg_start_time_shift=3,
            hsg_nominal_start_prob=1,
            hsg_mean_restart_time=12,
            hsg_restart_time_std=1,
            hsg_restart_time_shift=3,
            hsg_nominal_restart_prob=0.8,
        ),
        env_config=transit.EnvironmentConfiguration(
            current_velocity_component_from_north=config['currents_north'],
            current_velocity_component_from_east=config['currents_east'],
            wind_speed=config['wind_speed'],
            wind_direction=config['wind_direction'],
        ),
        sim_config=transit.SimulationConfiguration(
            initial_north_position_m=0,
            initial_east_position_m=0,
            initial_yaw_angle_rad=0,
            initial_forward_speed_m_per_s=0,
            initial_sideways_speed_m_per_s=0,
            initial_yaw_rate_rad_per_s=0,
            integration_step=config['integration_step'],
            simulation_time=config['max_sim_time'],
        ),
        risk_model_config=risks.RiskModelConfiguration(
            max_drift_time_s=config['max_sim_time'],
            risk_time_interval=config['time_interval'],
        ),
        environment=obstacles,
    )
    return setup
