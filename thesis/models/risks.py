"""
https://github.com/BorgeRokseth/drifting-grounding-risk-model -> risk_model.py

Provides classes to estimate drifting grounding risk either for a small time
interval or a prediction horizon (a set of adjoining small time intervals).

Only relevant parts are extracted from the external repository file.
"""

import numpy as np
from typing import NamedTuple
import shapely.geometry as geo
from seacharts.spatial import shapes

from thesis.models import events, transit


class RiskModelConfiguration(NamedTuple):
    max_drift_time_s: float
    risk_time_interval: float


class ShipPose:
    def __init__(self, north, east, heading_deg):
        self.north = north
        self.east = east
        self.heading_deg = heading_deg


class ScenarioAnalysisParameters(NamedTuple):
    main_engine_failure_rate: float
    main_engine_mean_start_time: float
    main_engine_start_time_std: float
    main_engine_start_time_shift: float
    main_engine_nominal_start_prob: float
    main_engine_mean_restart_time: float
    main_engine_restart_time_std: float
    main_engine_restart_time_shift: float
    main_engine_nominal_restart_prob: float
    genset_one_failure_rate: float
    genset_one_mean_start_time: float
    genset_one_start_time_std: float
    genset_one_start_time_shift: float
    genset_one_nominal_start_prob: float
    genset_one_mean_restart_time: float
    genset_one_restart_time_std: float
    genset_one_restart_time_shift: float
    genset_one_nominal_restart_prob: float
    genset_two_failure_rate: float
    genset_two_mean_start_time: float
    genset_two_start_time_std: float
    genset_two_start_time_shift: float
    genset_two_nominal_start_prob: float
    genset_two_mean_restart_time: float
    genset_two_restart_time_std: float
    genset_two_restart_time_shift: float
    genset_two_nominal_restart_prob: float
    hsg_failure_rate: float
    hsg_mean_start_time: float
    hsg_start_time_std: float
    hsg_start_time_shift: float
    hsg_nominal_start_prob: float
    hsg_mean_restart_time: float
    hsg_restart_time_std: float
    hsg_restart_time_shift: float
    hsg_nominal_restart_prob: float


class ScenarioProbabilitiesOutput(NamedTuple):
    pto_mode_scenarios: events.MachinerySystemOperatingMode
    mec_mode_scenarios: events.MachinerySystemOperatingMode
    pti_mode_scenarios: events.MachinerySystemOperatingMode


class RiskModelOutput(NamedTuple):
    probability_of_grounding_in_pto: float
    probability_of_grounding_in_mec: float
    probability_of_grounding_in_pti: float
    consequence_of_grounding: float


class GroundingRiskModel:
    def __init__(
        self,
        risk_model_config: RiskModelConfiguration,
        ttg_sim_config: transit.ShipConfiguration,
        sim_config: transit.SimulationConfiguration,
        env_config: transit.EnvironmentConfiguration,
        environment: geo.multipolygon.MultiPolygon,
        scenario_params: ScenarioAnalysisParameters,
    ):
        self.max_simulation_time = risk_model_config.max_drift_time_s
        self.risk_time_interval = risk_model_config.risk_time_interval
        self.ship_config = ttg_sim_config
        self.sim_config = sim_config
        self.env_config = env_config
        self.environment = environment
        self.scenario_params = scenario_params
        self.ttg_simulator = TimeToGroundingSimulator(
            max_simulation_time=self.max_simulation_time,
            ship_config=self.ship_config,
            simulation_config=self.sim_config,
            environment_config=self.env_config,
            environment=self.environment,
        )
        self.risk_model_output = None

    @property
    def probabilities(self) -> dict:
        return dict(
            pto=self.risk_model_output.probability_of_grounding_in_pto,
            mec=self.risk_model_output.probability_of_grounding_in_mec,
            pti=self.risk_model_output.probability_of_grounding_in_pti,
        )

    def set_initial_states(self, x, y, psi, speed):
        sim_config = transit.SimulationConfiguration(
            initial_north_position_m=x,
            initial_east_position_m=y,
            initial_yaw_angle_rad=psi,
            initial_forward_speed_m_per_s=speed,
            initial_sideways_speed_m_per_s=0,
            initial_yaw_rate_rad_per_s=0,
            integration_step=self.sim_config.integration_step,
            simulation_time=self.sim_config.simulation_time,
        )
        self.ttg_simulator.ship_model = transit.ShipModelWithoutPropulsion(
            ship_config=self.ttg_simulator.ship_config,
            environment_config=self.ttg_simulator.environment_config,
            simulation_config=sim_config,
        )

    def calculate_risk_output(self):
        time_to_grounding, consequence_of_grounding = (
            self.ttg_simulator.time_to_grounding()
        )
        scenario_analysis_output = self.scenario_analysis(
            available_recovery_time=time_to_grounding
        )
        self.risk_model_output = RiskModelOutput(
            probability_of_grounding_in_pto=scenario_analysis_output.pto_mode_scenarios.probability_of_grounding,
            probability_of_grounding_in_mec=scenario_analysis_output.mec_mode_scenarios.probability_of_grounding,
            probability_of_grounding_in_pti=scenario_analysis_output.pti_mode_scenarios.probability_of_grounding,
            consequence_of_grounding=consequence_of_grounding,
        )
        return self.risk_model_output

    def scenario_analysis(self, available_recovery_time: float):
        scenario = CaseStudyScenario(
            available_recovery_time=available_recovery_time,
            risk_time_interval=self.risk_time_interval,
            scenario_parameters=self.scenario_params,
        )
        return scenario.scenario_probabilities()


class CaseStudyScenario:
    def __init__(
        self,
        available_recovery_time: float,
        risk_time_interval: float,
        scenario_parameters: ScenarioAnalysisParameters,
    ) -> None:
        self.scenario_params = scenario_parameters
        self.risk_time_interval = risk_time_interval
        main_engine_stops = events.TriggeringEvent(
            rate_of_occurrence=self.scenario_params.main_engine_failure_rate,
            time_interval=self.risk_time_interval,
        )
        genset_one_stops = events.TriggeringEvent(
            rate_of_occurrence=self.scenario_params.genset_one_failure_rate,
            time_interval=self.risk_time_interval,
        )
        genset_two_stops = events.TriggeringEvent(
            rate_of_occurrence=self.scenario_params.genset_two_failure_rate,
            time_interval=self.risk_time_interval,
        )
        shaft_gen_stops = events.TriggeringEvent(
            rate_of_occurrence=self.scenario_params.hsg_failure_rate,
            time_interval=self.risk_time_interval,
        )

        loss_of_main_engine = events.LossOfPropulsionScenario(
            [main_engine_stops]
        )
        loss_of_both_gensets = events.LossOfPropulsionScenario(
            [genset_one_stops, genset_two_stops]
        )
        loss_of_shaft_gen = events.LossOfPropulsionScenario([shaft_gen_stops])

        start_main_engine_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.main_engine_mean_start_time,
            standard_deviation_time_to_restart=self.scenario_params.main_engine_start_time_std,
            time_shift_time_to_restart=self.scenario_params.main_engine_start_time_shift,
            nominal_success_probability=self.scenario_params.main_engine_nominal_start_prob,
        )
        start_main_engine = events.StartUpEvent(
            parameters=start_main_engine_params,
            time_available=available_recovery_time,
        )
        restart_main_engine_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.main_engine_mean_restart_time,
            standard_deviation_time_to_restart=self.scenario_params.main_engine_restart_time_std,
            time_shift_time_to_restart=self.scenario_params.main_engine_restart_time_shift,
            nominal_success_probability=self.scenario_params.main_engine_nominal_restart_prob,
        )
        restart_main_engine = events.StartUpEvent(
            parameters=restart_main_engine_params,
            time_available=available_recovery_time,
        )
        start_genset_one_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.genset_one_mean_start_time,
            standard_deviation_time_to_restart=self.scenario_params.genset_one_start_time_std,
            time_shift_time_to_restart=self.scenario_params.genset_one_start_time_shift,
            nominal_success_probability=self.scenario_params.genset_one_nominal_start_prob,
        )
        start_genset_one = events.StartUpEvent(
            parameters=start_genset_one_params,
            time_available=available_recovery_time,
        )
        start_genset_two_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.genset_two_mean_start_time,
            standard_deviation_time_to_restart=self.scenario_params.genset_two_start_time_std,
            time_shift_time_to_restart=self.scenario_params.genset_two_start_time_shift,
            nominal_success_probability=self.scenario_params.genset_two_nominal_start_prob,
        )
        start_genset_two = events.StartUpEvent(
            parameters=start_genset_two_params,
            time_available=available_recovery_time,
        )

        restart_genset_one_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.genset_one_mean_restart_time,
            standard_deviation_time_to_restart=self.scenario_params.genset_one_restart_time_std,
            time_shift_time_to_restart=self.scenario_params.genset_one_restart_time_shift,
            nominal_success_probability=self.scenario_params.genset_one_nominal_restart_prob,
        )
        restart_genset_one = events.StartUpEvent(
            parameters=restart_genset_one_params,
            time_available=available_recovery_time,
        )
        restart_genset_two_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.genset_two_mean_restart_time,
            standard_deviation_time_to_restart=self.scenario_params.genset_two_restart_time_std,
            time_shift_time_to_restart=self.scenario_params.genset_two_restart_time_shift,
            nominal_success_probability=self.scenario_params.genset_two_nominal_restart_prob,
        )
        restart_genset_two = events.StartUpEvent(
            parameters=restart_genset_two_params,
            time_available=available_recovery_time,
        )

        start_hsg_as_motor_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.hsg_mean_start_time,
            standard_deviation_time_to_restart=self.scenario_params.hsg_start_time_std,
            time_shift_time_to_restart=self.scenario_params.hsg_start_time_shift,
            nominal_success_probability=self.scenario_params.hsg_nominal_start_prob,
        )
        start_hsg_as_motor = events.StartUpEvent(
            parameters=start_hsg_as_motor_params,
            time_available=available_recovery_time,
        )

        restart_hsg_as_motor_params = events.StartUpEventParameters(
            mean_time_to_restart_s=self.scenario_params.hsg_mean_restart_time,
            standard_deviation_time_to_restart=self.scenario_params.hsg_restart_time_std,
            time_shift_time_to_restart=self.scenario_params.hsg_restart_time_shift,
            nominal_success_probability=self.scenario_params.hsg_nominal_restart_prob,
        )
        restart_hsg_as_motor = events.StartUpEvent(
            parameters=restart_hsg_as_motor_params,
            time_available=available_recovery_time,
        )

        restore_from_loss_of_main_engine_in_pto = (
            events.PowerRestorationEventTree(
                success_paths=[
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=restart_main_engine, occurs=True
                            ),
                        ]
                    ),
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=restart_main_engine, occurs=False
                            ),
                            events.PathElement(
                                event=start_genset_one, occurs=True
                            ),
                            events.PathElement(
                                event=start_hsg_as_motor, occurs=True
                            ),
                        ]
                    ),
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=restart_main_engine, occurs=False
                            ),
                            events.PathElement(
                                event=start_genset_one, occurs=False
                            ),
                            events.PathElement(
                                event=start_genset_two, occurs=True
                            ),
                            events.PathElement(
                                event=start_hsg_as_motor, occurs=True
                            ),
                        ]
                    ),
                ],
                time_to_grounding=available_recovery_time,
            )
        )
        restore_from_loss_of_main_engine_in_mec = (
            events.PowerRestorationEventTree(
                success_paths=[
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=restart_main_engine, occurs=True
                            ),
                        ]
                    ),
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=restart_main_engine, occurs=False
                            ),
                            events.PathElement(
                                event=start_hsg_as_motor, occurs=True
                            ),
                        ]
                    ),
                ],
                time_to_grounding=available_recovery_time,
            )
        )
        restore_from_loss_of_both_gensets_in_pti = (
            events.PowerRestorationEventTree(
                success_paths=[
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=start_main_engine, occurs=True
                            ),
                        ]
                    ),
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=start_main_engine, occurs=False
                            ),
                            events.PathElement(
                                event=restart_genset_one, occurs=True
                            ),
                        ]
                    ),
                    events.EventTreePath(
                        path=[
                            events.PathElement(
                                event=start_main_engine, occurs=False
                            ),
                            events.PathElement(
                                event=restart_genset_one, occurs=False
                            ),
                            events.PathElement(
                                event=restart_genset_one, occurs=True
                            ),
                        ]
                    ),
                ],
                time_to_grounding=available_recovery_time,
            )
        )
        restore_from_loss_of_hsg_in_pti = events.PowerRestorationEventTree(
            success_paths=[
                events.EventTreePath(
                    path=[
                        events.PathElement(
                            event=start_main_engine, occurs=True
                        ),
                    ]
                ),
                events.EventTreePath(
                    path=[
                        events.PathElement(
                            event=start_main_engine, occurs=False
                        ),
                        events.PathElement(
                            event=restart_hsg_as_motor, occurs=True
                        ),
                    ]
                ),
            ],
            time_to_grounding=available_recovery_time,
        )

        self.loss_of_main_engine_in_pto = events.Scenario(
            loss_scenario=loss_of_main_engine,
            restoration_scenario=restore_from_loss_of_main_engine_in_pto,
        )
        self.loss_of_main_engine_in_mec = events.Scenario(
            loss_scenario=loss_of_main_engine,
            restoration_scenario=restore_from_loss_of_main_engine_in_mec,
        )
        self.loss_of_both_gensets_in_pti = events.Scenario(
            loss_scenario=loss_of_both_gensets,
            restoration_scenario=restore_from_loss_of_both_gensets_in_pti,
        )
        self.loss_of_hsg_in_pti = events.Scenario(
            loss_scenario=loss_of_shaft_gen,
            restoration_scenario=restore_from_loss_of_hsg_in_pti,
        )

    def scenario_probabilities(self):
        return ScenarioProbabilitiesOutput(
            pto_mode_scenarios=events.MachinerySystemOperatingMode(
                possible_scenarios=[self.loss_of_main_engine_in_pto]
            ),
            mec_mode_scenarios=events.MachinerySystemOperatingMode(
                possible_scenarios=[self.loss_of_main_engine_in_mec]
            ),
            pti_mode_scenarios=events.MachinerySystemOperatingMode(
                possible_scenarios=[
                    self.loss_of_both_gensets_in_pti,
                    self.loss_of_hsg_in_pti,
                ]
            ),
        )


class TimeToGroundingSimulator:
    """Simulate a ship drifting from given initial states until a grounding occurs
    or the maximum simulation time has elapsed.
    """

    def __init__(
        self,
        environment: geo.multipolygon.MultiPolygon,
        max_simulation_time: float,
        ship_config: transit.ShipConfiguration,
        simulation_config: transit.SimulationConfiguration,
        environment_config: transit.EnvironmentConfiguration,
    ):
        """Set up simulation.

        args:
        - initial_states (MotionStateInput): See MotionStateInput-class
        - max_simulation_time (float): Number of seconds after which to terminate
        simulation if
        grounding has not occurred.
        """
        self.max_sim_time = max_simulation_time
        self.environment = environment
        self.ship_config = ship_config
        self.simulation_config = simulation_config
        self.environment_config = environment_config
        self.drifting_ship_positions = []
        self.ship_model = transit.ShipModelWithoutPropulsion(
            ship_config=self.ship_config,
            environment_config=self.environment_config,
            simulation_config=self.simulation_config,
        )

    def time_to_grounding(self):
        """Find the time it will take to ground and the magnitude of the consequence
        of grounding based on the speed of impact and the character of the shore
        (whether or not the ship hits infrastructure such as fish-farms. If the
        simulation is terminated before grounding occurs, the consequence of
        grounding
        is calculated based on the speed of the ship when the simulation is
        terminated
        and the cheapest shore character.

        returns:
        - time_to_grounding (float): Number of seconds it takes before the ship
        grounds
        - consequence_of_grounding (float): The cost of the impact.
        """
        consequence_of_grounding = 2000
        # Necessary to set initial states?
        grounded = False
        ship_pose_interval = 25
        time_since_last_ship_pose = 0
        while (
            self.ship_model.int.time <= self.ship_model.int.sim_time
            and not grounded
        ):
            self.ship_model.update_differentials()
            self.ship_model.integrate_differentials()
            self.ship_model.store_simulation_data()
            if time_since_last_ship_pose >= ship_pose_interval:
                self.drifting_ship_positions.append(
                    ShipPose(
                        north=self.ship_model.north,
                        east=self.ship_model.east,
                        heading_deg=self.ship_model.yaw_angle * 180 / np.pi,
                    )
                )
                time_since_last_ship_pose = 0
            self.ship_model.int.next_time()
            time_since_last_ship_pose += self.ship_model.int.dt
            grounded = self.check_if_grounded(
                self.ship_model.north,
                self.ship_model.east,
                self.ship_model.yaw_angle,
            )
        time_to_grounding = self.ship_model.int.time
        return time_to_grounding, consequence_of_grounding

    def check_if_grounded(self, north, east, yaw):
        ship = shapes.Ship(east, north, yaw, in_degrees=False).geometry
        return ship.intersects(self.environment)
