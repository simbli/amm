"""
https://github.com/BorgeRokseth/drifting-grounding-risk-model -> scenarios.py

Provides classes to simulate event trees and loss of propulsion scenarios.

Only relevant parts are extracted from the external repository file.
"""

from typing import NamedTuple, List
from xmlrpc.client import Boolean
import numpy as np
from scipy import stats


class StartUpEventParameters(NamedTuple):
    mean_time_to_restart_s: float
    standard_deviation_time_to_restart: float
    time_shift_time_to_restart: float
    nominal_success_probability: float


class StartUpEvent:
    def __init__(self, parameters: StartUpEventParameters, time_available):
        self.mu = parameters.mean_time_to_restart_s
        self.sigma = parameters.standard_deviation_time_to_restart
        self.startup_time_distribution = stats.lognorm(
            scale=parameters.mean_time_to_restart_s,
            s=np.sqrt(parameters.standard_deviation_time_to_restart),
            loc=parameters.time_shift_time_to_restart,
        )
        self.nominal_success_probability = (
            parameters.nominal_success_probability
        )
        self.probability = self.probability_calculation(time=time_available)

    def probability_calculation(self, time):
        return (
            self.nominal_success_probability
            * self.startup_time_distribution.cdf(time)
        )

    def update_probability(self, time):
        self.probability = self.probability_calculation(time)


class PathElement:
    def __init__(self, event: StartUpEvent, occurs: Boolean) -> None:
        self.event = event
        self.occurs = occurs

    def update_probability(self, time):
        self.event.update_probability(time=time)


class EventTreePath:
    def __init__(self, path: List[PathElement]) -> None:
        self.path = path
        self.path_probability = 1

    def update_path_probability(self, new_available_time) -> None:
        path_probability = 1
        for path_element in self.path:
            path_element.update_probability(time=new_available_time)
            if path_element.occurs:
                path_probability = (
                    path_probability * path_element.event.probability
                )
            else:
                path_probability = path_probability * (
                    1 - path_element.event.probability
                )
        self.path_probability = path_probability


class PowerRestorationEventTree:
    def __init__(self, success_paths: List[EventTreePath], time_to_grounding):
        self.success_paths = success_paths
        self.probability = self.probability_of_success(
            new_available_time=time_to_grounding
        )

    def _update_success_paths(self, new_available_time):
        for path in self.success_paths:
            path.update_path_probability(new_available_time)

    def probability_of_success(self, new_available_time) -> float:
        probability_of_success = 0
        self._update_success_paths(new_available_time=new_available_time)
        for path in self.success_paths:
            probability_of_success = (
                probability_of_success + path.path_probability
            )
        return probability_of_success


class TriggeringEvent:
    def __init__(self, rate_of_occurrence, time_interval):
        self.rate = rate_of_occurrence
        self.dt = time_interval
        self.probability = self.probability_of_occurrence()

    def probability_of_occurrence(self):
        return 1 - np.exp(-self.rate * self.dt)


class LossOfPropulsionScenario:
    """
    A LOPP-scenario is described as a minimial cutset of triggering events.
    """

    def __init__(self, triggering_events: List[TriggeringEvent]):
        list_of_triggering_event_probabilities = []
        for event in triggering_events:
            list_of_triggering_event_probabilities.append(event.probability)
        self.probability = np.prod(list_of_triggering_event_probabilities)


class Scenario:
    def __init__(
        self,
        loss_scenario: LossOfPropulsionScenario,
        restoration_scenario: PowerRestorationEventTree,
    ):
        self.loss_scenario = loss_scenario
        self.restoration_scenario = restoration_scenario


class MachinerySystemOperatingMode:
    def __init__(self, possible_scenarios: List[Scenario]):
        self.scenarios = possible_scenarios
        self.probability_of_grounding = self.probability_calculation()

    def probability_calculation(self):
        prod = 1
        for s in self.scenarios:
            prob_of_grounding_given_loss = (
                1 - s.restoration_scenario.probability
            )
            prod = (
                prod
                - prob_of_grounding_given_loss * s.loss_scenario.probability
            )
        return 1 - prod
