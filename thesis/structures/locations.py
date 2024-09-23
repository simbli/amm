"""
Contains the Location, ShipPose, and Waypoint data classes.
"""

from dataclasses import dataclass


@dataclass
class Location:
    """
    Data class for a spatial location with X and Y coordinates.
    """

    x: float
    y: float

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y


Locations = list[Location]


@dataclass
class ShipPose(Location):
    """
    Data class for a ship pose (location with orientation / heading / yaw).
    """

    heading: float

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.heading


Trajectory = list[ShipPose]
Trajectories = list[Trajectory]


@dataclass
class Waypoint(Location):
    """
    Data class for a waypoint location with a machinery system operation mode.
    """

    mso: str = ''
    length: float = 0.0

    @property
    def as_tuple(self) -> tuple[float, float, str, float]:
        return self.x, self.y, self.mso, self.length


Path = list[Waypoint]
Paths = list[Path]
