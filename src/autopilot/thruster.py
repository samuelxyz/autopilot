import numpy as np

from autopilot import actor
from autopilot import state_def as sd


class Thruster(actor.Actor):
    def command(self, command):
        pass

    def get_wrench(self, state, mass, local_inertia_matrix):
        pass


# 6 thrusters (+/- xyz) that remain aligned to world coordinates regardless of ship orientation
class Thrusters_Floating_6(Thruster):
    def __init__(self, max_thrust=np.inf, min_thrust=0.0):
        super().__init__()
        self.max_thrust = max_thrust  # by axis
        self.min_thrust = min_thrust
        self.thrusts = np.zeros(3)

    def command(self, thrusts_xyz):
        """Set the commanded thrust in each axis (xyz), 3 component array. Will be clipped between the thruster's min and max accel. If less than half the min accel in any axis, will be clipped to zero."""
        clipped = np.clip(
            np.abs(thrusts_xyz), self.min_thrust, self.max_thrust
        ) * np.sign(thrusts_xyz)
        nonzero = np.abs(thrusts_xyz) > self.min_thrust / 2
        self.thrusts = clipped * nonzero

    def get_wrench(self, state, mass, local_inertia_matrix):
        return sd.make_wrench(lin=self.thrusts)


# 6 thrusters (+/- xyz) fixed to the ship's body frame. Can only be on or off
class Thrusters_Fixed_Binary_6(Thruster):
    def __init__(self, thrust_spec):
        """thrust_spec is the acceleration produced by a thruster when on. off would be 0"""
        super().__init__()
        self.thrust_spec = thrust_spec
        self.thrusts = np.zeros(3)

    def command(self, thrusts_xyz_body):
        """thrusts_xyz_body: 3 components, positive/negative/zero according to np.sign()"""
        self.thrusts = np.sign(thrusts_xyz_body) * self.thrust_spec

    def get_wrench(self, state, mass, local_inertia_matrix):
        return sd.make_wrench(
            lin=sd.rotate_vector(sd.get_orient_q(state), self.thrusts)
        )
