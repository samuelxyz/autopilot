import numpy as np

from autopilot import thruster
from autopilot import state_def as sd


class Controller:
    """Dummy/parent controller class with no implemented functionality."""

    def __init__(self, actuator: thruster.Thruster = None):
        self.actuator = actuator

    def update(self, ship_state, dt):
        pass


class PID(Controller):
    """Simple PID controller, can operate array-wise according to the shape parameter."""

    def __init__(self, setpoint, gains, shape, thruster: thruster.Thruster):
        """setpoint: vec3
        gains: vec3"""
        super().__init__(thruster)
        self.setpoint = setpoint
        self.k_P, self.k_I, self.k_D = gains
        self.output = np.zeros(shape)
        self.last_error = np.zeros(shape)
        self.integral = np.zeros(shape)

    def update(self, ship_state, dt):
        error = self.setpoint - ship_state[sd.POS]

        p_term = self.k_P * error

        # Triangle rule
        self.integral += (error + self.last_error) / 2 * dt
        i_term = self.k_I * self.integral

        # Finite difference
        d_term = self.k_D * (error - self.last_error) / dt
        self.last_error = error

        self.output = p_term + i_term + d_term

        self.actuator.command(self.output)


class Binary_Controller_3(Controller):
    """xyz translational controller using body-fixed thrusters that can be on/off only.
    For use with Thrusters_Fixed_Binary_6."""

    def __init__(self, setpoint, accel, thruster: thruster.Thrusters_Fixed_Binary_6):
        """acc: the acceleration produced by firing one thruster"""
        super().__init__(thruster)
        self.setpoint = setpoint
        self.accel = accel

        # fudgy factor to discourage overshooting
        self.margin = 1.2

        # positional deadband
        self.position_deadband = 0.05
        self.velocity_deadband = 0.1

    def update(self, ship_state, dt):
        ship_state_without_angvel = ship_state.copy()
        ship_state_without_angvel[sd.ANGVEL] = 0
        relative_setpoint = sd.change_to_frame(ship_state_without_angvel, self.setpoint)
        command = np.zeros(3)

        for axis in range(3):
            target_pos = relative_setpoint[sd.POS.start + axis]
            my_vel = -relative_setpoint[sd.VEL.start + axis]

            # short circuit: deadband behavior
            if np.abs(target_pos) < self.position_deadband:
                if np.abs(my_vel) < self.velocity_deadband:
                    # no action needed?
                    command[axis] = 0
                    continue

            # short circuit: if we're already moving the wrong way, stop it
            if target_pos * my_vel < 0.0:
                command[axis] = np.sign(target_pos)
                continue

            # if we're moving the right way, decide whether to speed up or slow down
            # v^2 - v_0^2 = 2ad
            stopping_distance = my_vel**2 / (2 * self.accel)
            if abs(target_pos) > stopping_distance * self.margin:
                # what happens if we speed up during this timestep?
                new_speed = np.abs(my_vel) + self.accel * dt
                new_stopping_distance = new_speed**2 / (2 * self.accel)
                if abs(target_pos) > new_stopping_distance * self.margin:
                    # we can speed up and it wont cause overshooting
                    command[axis] = np.sign(target_pos)
                else:
                    command[axis] = 0
            else:
                # slow down
                command[axis] = -np.sign(target_pos)

        self.actuator.command(command)
