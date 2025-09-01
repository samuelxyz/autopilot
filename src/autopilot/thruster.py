import numpy as np
import quaternion as quat

import actor
import state_def as sd


class Thruster(actor.Actor):
    def command(self, command):
        pass

    def get_accel(self, state, mass, local_inertia_matrix):
        pass


# 6 thrusters (+/- xyz) that remain aligned to world coordinates regardless of ship orientation
class Thrusters_Floating_6(Thruster):
    def __init__(self, max_accel=np.inf, min_accel=0.0):
        super().__init__()
        self.max_accel = max_accel  # by axis
        self.min_accel = min_accel
        self.thrusts = np.zeros(3)

    def command(self, accel_xyz):
        """Set the commanded thrust in each axis (xyz), 3 component array. Will be clipped between the thruster's min and max accel. If less than half the min accel in any axis, will be clipped to zero."""
        clipped = np.clip(np.abs(accel_xyz), self.min_accel, self.max_accel) * np.sign(
            accel_xyz
        )
        nonzero = np.abs(accel_xyz) > self.min_accel / 2
        self.thrusts = clipped * nonzero

    def get_accel(self, state, mass, local_inertia_matrix):
        result = np.zeros((sd.QDOT_N,))
        result[sd.LIN] = self.thrusts
        return result


# 6 thrusters (+/- xyz) fixed to the ship's body frame. Can only be on or off
class Thrusters_Fixed_Binary_6(Thruster):
    def __init__(self, accel):
        """accel is the acceleration produced by a thruster when on. off would be 0"""
        super().__init__()
        self.accel = accel
        self.thrusts = np.zeros(3)

    def command(self, accel_xyz_body):
        """accel_xyz_body: 3 components, positive/negative/zero according to np.sign()"""
        self.thrusts = np.sign(accel_xyz_body) * self.accel

    def get_accel(self, state, mass, local_inertia_matrix):
        result = np.zeros(sd.QDOT_N)
        result[sd.LIN] = quat.rotate_vectors(sd.get_orient_q(state), self.thrusts)
        return result
