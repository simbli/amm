"""
Contains the Scenario class and the select function for selecting a scenario.
"""

_SCENARIOS = {
    '2020mpc': {},
    '2021enc': {},
    '2022pso': {},
    '2023amm': {
        10: ['grounding'],
        12: ['grounding+winds'],
        14: ['winds+currents'],
        15: ['winds+currents'],
        16: ['winds+currents'],
    },
    '2023rsc': {},
}


class Scenario:
    """
    Simulates specific scenario conditions for article figures.
    """

    def __init__(self, figure: int, variant: str):
        self.figure = figure
        self.variant = variant
        self.ocean_currents = 0
        self.wind_velocity = 0
        self.wind_direction = 0
        if 'winds' in self.variant:
            self.wind_direction = -110
            if 'grounding' in self.variant:
                self.wind_direction = -111
            self.wind_velocity = 10
            if 'currents' in self.variant:
                self.ocean_currents = 1

    @property
    def disturbances(self) -> tuple[float, float, float]:
        return self.wind_direction, self.wind_velocity, self.ocean_currents


def select(article: str, figure: int) -> list[Scenario]:
    """Returns the scenarios specified for the given article figure."""
    selection = _SCENARIOS[article].get(figure, [])
    return [Scenario(figure, v) for v in selection]
