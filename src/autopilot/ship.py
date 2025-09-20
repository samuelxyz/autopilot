# Ship
# Contains a 6-DOF rigid body simulation that will be controlled by various actuators and controllers
# RK4 time-stepping

import numpy as np
from autopilot import actor
from autopilot import controller
from autopilot import state_def as sd
from autopilot import sensor


class Ship:
    def __init__(
        self,
        state,
        mass,
        inertia_matrix,
        controller_: controller.Controller = None,
        state_det_system: sensor.SensorSystem = None,
        actors: list[actor.Actor] = [],
    ):
        self.state = np.asarray(state)  # current state, updated every timestep. q, qdot
        # self.state_history = ...
        self.mass = mass  # something
        self.inertia_matrix = inertia_matrix  # body frame
        self.controller = controller_
        if self.controller is None:
            self.controller = controller.Controller()
        self.state_det_system = state_det_system
        if self.state_det_system is None:
            self.state_det_system = sensor.SensorSystem(self.state)
        self.actors = actors
        self.setpoint = np.ones(3)

    def update(self, start_time, dt):
        """Advance the ship simulation by a timestep of length dt, including running
        sensors/controllers and then updating state.

        Parameters:
            start_time: time at beginning of timestep
            dt: length of timestep
        """
        # Physics
        self.state = sd.RK4_newtoneuler_step(
            self.state, self.mass, self.inertia_matrix, dt, self.calc_total_wrench
        )

        # Predict
        self.state_det_system.predict_step(
            self.mass, self.inertia_matrix, dt, self.calc_total_wrench
        )

        # Sense
        self.state_det_system.sense(self.state, start_time)
        self.state_estimate = self.state_det_system.get_current_state_estimate()

        # Control
        if self.controller is not None:
            self.controller.update(
                self.state_estimate[0], self.mass, self.inertia_matrix, dt
            )

    def calc_total_wrench(self, state) -> np.ndarray:
        """Calculates the force and moment (total shape (6,))
        from ship properties and the given state, in the world frame"""
        result = np.zeros((sd.WRENCH_N,))
        # rotation_matrix = quat.as_rotation_matrix(sd.get_orient(self.state))
        # world_inertia_matrix = rotation_matrix @ self.inertia_matrix @ rotation_matrix.T
        for actor_ in self.actors:
            result += actor_.get_wrench(state, self.mass, self.inertia_matrix)
        return result

    def calc_apparent_wrench(self, state):
        """Like calc_total_wrench but excludes gravity. Intended for reporting/logging purposes."""
        result = np.zeros((sd.WRENCH_N,))
        for actor_ in self.actors:
            if actor_.is_apparent:
                result += actor_.get_wrench(state, self.mass, self.inertia_matrix)
        return result
