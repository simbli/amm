"""
Contains the Simulator class for simulating disturbances and path following.
"""

from thesis.simulation import controller, scenarios


class Simulator:
    """
    Contains scenario disturbances and the ship controller for path following.
    """

    def __init__(self, article: str, figure: int):
        self.scenarios = scenarios.select(article, figure)
        self.controller = controller.HeadingController()

    @property
    def disturbances(self) -> tuple[float, float, float]:
        return self.scenarios[0].disturbances
