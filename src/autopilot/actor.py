# Actor
# Something that could apply forces on the ship: thrusters, gravity, aerodynamics, etc
import numpy as np

from autopilot import state_def as sd

gravitational_constant_si = 6.6743e-11
GM_Earth = 3.98600442e14
GM_Sun = 1.3271244002e20
GM_Moon = 4.902800118e12


class Actor:
    def __init__(self, is_apparent=True):
        self.is_apparent = is_apparent

    def get_accel(self, state, mass, local_inertia_matrix):
        return np.zeros((sd.QDOT_N,))


class Gravity(Actor):
    def __init__(self, pos, GM):
        super().__init__(is_apparent=False)
        self.pos = np.asarray(pos)
        self.GM = GM

    def get_accel(self, state, mass, inertia_matrix, min_radius=1):
        """min_radius is to avoid gravitational singularities"""
        r = state[sd.POS] - self.pos
        result = np.zeros((sd.QDOT_N,))
        if np.linalg.norm(r) < min_radius:
            return result  # hardcoded to avoid bugginess
        result[sd.LIN] = -self.GM * r / np.linalg.norm(r) ** 3
        return result


class Drag_Equation(Actor):
    """Actor that simulates simple aerodynamic drag parallel to airspeed and scaling with its square."""

    def __init__(self, drag_coeff, cs_area, state_to_density, state_to_airspeed):
        super().__init__()
        self.drag_coeff = drag_coeff
        self.cs_area = cs_area  # cross section
        self.state_to_density = state_to_density
        self.state_to_airspeed = state_to_airspeed

    def get_accel(self, state, mass, inertia_matrix):
        v = self.state_to_airspeed(state)
        rho = self.state_to_density(state)
        force = -0.5 * rho * v * np.linalg.norm(v) * self.drag_coeff * self.cs_area
        return sd.make_qdot(lin=force / mass)


class Lift_Equation(Actor):
    """Actor that simulates simple aerodynamic lift perpendicular to airspeed and scaling with its square.
    Lift is in the plane spanned by the airspeed and ship z-vector (belly direction).
    You will probably want to manually update the lift coefficient.
    """

    def __init__(self, lift_coeff, wing_area, state_to_density, state_to_airspeed):
        super().__init__()
        self.lift_coeff = lift_coeff
        self.wing_area = wing_area  # cross section
        self.state_to_density = state_to_density
        self.state_to_airspeed = state_to_airspeed

    def get_accel(self, state, mass, inertia_matrix):
        v = self.state_to_airspeed(state)
        rho = self.state_to_density(state)
        belly_dir = sd.rotate_vector(sd.get_orient_q(state), (0, 0, 1))
        try:
            freestream_right_dir = sd.normalize_or_err(np.cross(belly_dir, v))
            lift_dir = sd.normalize_or_err(np.cross(freestream_right_dir, v))
        except ValueError:
            return sd.make_qdot()
        force = 0.5 * rho * np.dot(v, v) * self.lift_coeff * self.wing_area * lift_dir
        return sd.make_qdot(lin=force / mass)
