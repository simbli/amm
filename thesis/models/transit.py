"""
https://github.com/BorgeRokseth/ship_in_transit_simulator -> models.py

This module provides classes that that can be used to setup and
run simulation models of a ship in transit.

Only relevant parts are extracted from the external repository file.
"""

import numpy as np
from collections import defaultdict
from typing import NamedTuple


class ShipConfiguration(NamedTuple):
    dead_weight_tonnage: float
    coefficient_of_deadweight_to_displacement: float
    bunkers: float
    ballast: float
    length_of_ship: float
    width_of_ship: float
    added_mass_coefficient_in_surge: float
    added_mass_coefficient_in_sway: float
    added_mass_coefficient_in_yaw: float
    mass_over_linear_friction_coefficient_in_surge: float
    mass_over_linear_friction_coefficient_in_sway: float
    mass_over_linear_friction_coefficient_in_yaw: float
    nonlinear_friction_coefficient__in_surge: float
    nonlinear_friction_coefficient__in_sway: float
    nonlinear_friction_coefficient__in_yaw: float


class EnvironmentConfiguration(NamedTuple):
    current_velocity_component_from_north: float
    current_velocity_component_from_east: float
    wind_speed: float
    wind_direction: float


class SimulationConfiguration(NamedTuple):
    initial_north_position_m: float
    initial_east_position_m: float
    initial_yaw_angle_rad: float
    initial_forward_speed_m_per_s: float
    initial_sideways_speed_m_per_s: float
    initial_yaw_rate_rad_per_s: float
    integration_step: float
    simulation_time: float


class BaseShipModel:
    def __init__(
        self,
        ship_config: ShipConfiguration,
        simulation_config: SimulationConfiguration,
        environment_config: EnvironmentConfiguration,
    ):
        payload = 0.9 * (ship_config.dead_weight_tonnage - ship_config.bunkers)
        lsw = (
            ship_config.dead_weight_tonnage
            / ship_config.coefficient_of_deadweight_to_displacement
            - ship_config.dead_weight_tonnage
        )
        self.mass = lsw + payload + ship_config.bunkers + ship_config.ballast

        self.l_ship = ship_config.length_of_ship  # 80
        self.w_ship = ship_config.width_of_ship  # 16.0
        self.x_g = 0
        self.i_z = self.mass * (self.l_ship**2 + self.w_ship**2) / 12

        # zero-frequency added mass
        self.x_du, self.y_dv, self.n_dr = self.set_added_mass(
            ship_config.added_mass_coefficient_in_surge,
            ship_config.added_mass_coefficient_in_sway,
            ship_config.added_mass_coefficient_in_yaw,
        )

        self.t_surge = (
            ship_config.mass_over_linear_friction_coefficient_in_surge
        )
        self.t_sway = ship_config.mass_over_linear_friction_coefficient_in_sway
        self.t_yaw = ship_config.mass_over_linear_friction_coefficient_in_yaw
        self.ku = (
            ship_config.nonlinear_friction_coefficient__in_surge
        )  # 2400.0  # non-linear friction coeff in surge
        self.kv = (
            ship_config.nonlinear_friction_coefficient__in_sway
        )  # 4000.0  # non-linear friction coeff in sway
        self.kr = (
            ship_config.nonlinear_friction_coefficient__in_yaw
        )  # 400.0  # non-linear friction coeff in yaw

        # Environmental conditions
        self.vel_c = np.array(
            [
                environment_config.current_velocity_component_from_north,
                environment_config.current_velocity_component_from_east,
                0.0,
            ]
        )
        self.wind_dir = environment_config.wind_direction
        self.wind_speed = environment_config.wind_speed

        # Initialize states
        self.north = simulation_config.initial_north_position_m
        self.east = simulation_config.initial_east_position_m
        self.yaw_angle = simulation_config.initial_yaw_angle_rad
        self.forward_speed = simulation_config.initial_forward_speed_m_per_s
        self.sideways_speed = simulation_config.initial_sideways_speed_m_per_s
        self.yaw_rate = simulation_config.initial_yaw_rate_rad_per_s

        # Initialize differentials
        self.d_north = 0
        self.d_east = 0
        self.d_yaw = 0
        self.d_forward_speed = 0
        self.d_sideways_speed = 0
        self.d_yaw_rate = 0

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Setup wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = (
            self.w_ship * self.h_f
        )  # Projected are from the front
        self.proj_area_l = (
            self.l_ship * self.h_s
        )  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.016  # 0.08

    def set_added_mass(self, surge_coeff, sway_coeff, yaw_coeff):
        """Sets the added mass in surge due to surge motion, sway due
        to sway motion and yaw due to yaw motion according to given coeffs.

        args:
            surge_coeff (float): Added mass coefficient in surge direction due to surge motion
            sway_coeff (float): Added mass coefficient in sway direction due to sway motion
            yaw_coeff (float): Added mass coefficient in yaw direction due to yaw motion
        returns:
            x_du (float): Added mass in surge
            y_dv (float): Added mass in sway
            n_dr (float): Added mass in yaw
        """
        x_du = self.mass * surge_coeff
        y_dv = self.mass * sway_coeff
        n_dr = self.i_z * yaw_coeff
        return x_du, y_dv, n_dr

    def get_wind_force(self):
        """This method calculates the forces due to the relative
        wind speed, acting on the ship in surge, sway and yaw
        direction.

        :return: Wind force acting in surge, sway and yaw
        """
        uw = self.wind_speed * np.cos(self.wind_dir - self.yaw_angle)
        vw = self.wind_speed * np.sin(self.wind_dir - self.yaw_angle)
        u_rw = uw - self.forward_speed
        v_rw = vw - self.sideways_speed
        gamma_rw = -np.arctan2(v_rw, u_rw)
        wind_rw2 = u_rw**2 + v_rw**2
        c_x = -self.cx * np.cos(gamma_rw)
        c_y = self.cy * np.sin(gamma_rw)
        c_n = self.cn * np.sin(2 * gamma_rw)
        tau_coeff = 0.5 * self.rho_a * wind_rw2
        tau_u = tau_coeff * c_x * self.proj_area_f
        tau_v = tau_coeff * c_y * self.proj_area_l
        tau_n = tau_coeff * c_n * self.proj_area_l * self.l_ship
        return np.array([tau_u, tau_v, tau_n])

    def three_dof_kinematics(self):
        """Updates the time differientials of the north position, east
        position and yaw angle. Should be called in the simulation
        loop before the integration step.
        """
        vel = np.array(
            [self.forward_speed, self.sideways_speed, self.yaw_rate]
        )
        dx = np.dot(self.rotation(), vel)
        self.d_north = dx[0]
        self.d_east = dx[1]
        self.d_yaw = dx[2]

    def rotation(self):
        """Specifies the rotation matrix for rotations about the z-axis, such that
        "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        """
        return np.array(
            [
                [np.cos(self.yaw_angle), -np.sin(self.yaw_angle), 0],
                [np.sin(self.yaw_angle), np.cos(self.yaw_angle), 0],
                [0, 0, 1],
            ]
        )

    def mass_matrix(self):
        return np.array(
            [
                [self.mass + self.x_du, 0, 0],
                [0, self.mass + self.y_dv, self.mass * self.x_g],
                [0, self.mass * self.x_g, self.i_z + self.n_dr],
            ]
        )

    def coriolis_matrix(self):
        return np.array(
            [
                [
                    0,
                    0,
                    -self.mass
                    * (self.x_g * self.yaw_rate + self.sideways_speed),
                ],
                [0, 0, self.mass * self.forward_speed],
                [
                    self.mass
                    * (self.x_g * self.yaw_rate + self.sideways_speed),
                    -self.mass * self.forward_speed,
                    0,
                ],
            ]
        )

    def coriolis_added_mass_matrix(self, u_r, v_r):
        return np.array(
            [
                [0, 0, self.y_dv * v_r],
                [0, 0, -self.x_du * u_r],
                [-self.y_dv * v_r, self.x_du * u_r, 0],
            ]
        )

    def linear_damping_matrix(self):
        return np.array(
            [
                [self.mass / self.t_surge, 0, 0],
                [0, self.mass / self.t_sway, 0],
                [0, 0, self.i_z / self.t_yaw],
            ]
        )

    def non_linear_damping_matrix(self):
        return np.array(
            [
                [self.ku * self.forward_speed, 0, 0],
                [0, self.kv * self.sideways_speed, 0],
                [0, 0, self.kr * self.yaw_rate],
            ]
        )

    def three_dof_kinetics(self, *args, **kwargs):
        """Calculates accelerations of the ship, as a funciton
        of thrust-force, rudder angle, wind forces and the
        states in the previous time-step.
        """
        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])

        # assembling state vector
        vel = np.array(
            [self.forward_speed, self.sideways_speed, self.yaw_rate]
        )

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(
                self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c
            )
            - np.dot(
                self.linear_damping_matrix()
                + self.non_linear_damping_matrix(),
                vel - v_c,
            )
            - wind_force
            + wave_force,
        )
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def update_differentials(self, *args, **kwargs):
        """This method should be called in the simulation loop. It will
        update the full differential equation of the ship.
        """
        self.three_dof_kinematics()
        self.three_dof_kinetics()

    def integrate_differentials(self):
        """Integrates the differential equation one time step ahead using
        the euler intgration method with parameters set in the
        int-instantiation of the "EulerInt"-class.
        """
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(
            x=self.forward_speed, dx=self.d_forward_speed
        )
        self.sideways_speed = self.int.integrate(
            x=self.sideways_speed, dx=self.d_sideways_speed
        )
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)


class ShipModelWithoutPropulsion(BaseShipModel):
    """Creates a ship model object that can be used to simulate a ship drifting freely

    The model contains the following states:
    - North position of ship
    - East position of ship
    - Yaw angle (relative to north axis)
    - Surge velocity (forward)
    - Sway velocity (sideways)
    - Yaw rate

    Simulation results are stored in the instance variable simulation_results
    """

    def __init__(
        self,
        ship_config: ShipConfiguration,
        environment_config: EnvironmentConfiguration,
        simulation_config: SimulationConfiguration,
    ):
        super().__init__(ship_config, simulation_config, environment_config)
        self.simulation_results = defaultdict(list)

    def store_simulation_data(self):
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(
            self.yaw_angle * 180 / np.pi
        )
        self.simulation_results['forward speed[m/s]'].append(
            self.forward_speed
        )
        self.simulation_results['sideways speed [m/s]'].append(
            self.sideways_speed
        )
        self.simulation_results['yaw rate [deg/sec]'].append(
            self.yaw_rate * 180 / np.pi
        )
        self.simulation_results['wind speed [m/sec]'].append(self.wind_speed)


class EulerInt:
    """Provides methods relevant for using the
    Euler method to integrate an ODE.

    Usage:

    int=EulerInt()
    while int.time <= int.sim_time:
        dx = f(x)
        int.integrate(x,dx)
        int.next_time
    """

    def __init__(self):
        self.dt = 0.01
        self.sim_time = 10
        self.time = 0.0
        self.times = []
        self.global_times = []

    def set_dt(self, val):
        """Sets the integrator step length"""
        self.dt = val

    def set_sim_time(self, val):
        """Sets the upper time integration limit"""
        self.sim_time = val

    def set_time(self, val):
        """Sets the time variable to "val" """
        self.time = val

    def next_time(self, time_shift=0):
        """Increment the time variable to the next time instance
        and store in an array
        """
        self.time = self.time + self.dt
        self.times.append(self.time)
        self.global_times.append(self.time + time_shift)

    def integrate(self, x, dx):
        """Performs the Euler integration step"""
        return x + dx * self.dt
