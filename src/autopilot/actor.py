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

    def get_wrench(self, state, mass, inertia_matrix_body):
        return np.zeros((sd.WRENCH_N,))


class Gravity(Actor):
    def __init__(self, pos, GM):
        super().__init__(is_apparent=False)
        self.pos = np.asarray(pos)
        self.GM = GM

    def get_wrench(self, state, mass, inertia_matrix_body, min_radius=1):
        """min_radius is to avoid gravitational singularities"""
        r = state[sd.POS] - self.pos
        if np.linalg.norm(r) < min_radius:
            return sd.make_wrench()  # return zero instead of a huge/infinite wrench
        force = -self.GM * mass * r / np.linalg.norm(r) ** 3
        return sd.make_wrench(lin=force)


class Drag_Equation(Actor):
    """Actor that simulates simple aerodynamic drag parallel to airspeed and scaling with its square."""

    def __init__(self, drag_coeff, cs_area, state_to_density, state_to_airspeed):
        super().__init__()
        self.drag_coeff = drag_coeff
        self.cs_area = cs_area  # cross section
        self.state_to_density = state_to_density
        self.state_to_airspeed = state_to_airspeed

    def get_wrench(self, state, mass, inertia_matrix_body):
        v = self.state_to_airspeed(state)
        rho = self.state_to_density(state)
        force = -0.5 * rho * v * np.linalg.norm(v) * self.drag_coeff * self.cs_area
        return sd.make_wrench(lin=force)


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

    def get_wrench(self, state, mass, inertia_matrix_body):
        v = self.state_to_airspeed(state)
        rho = self.state_to_density(state)
        belly_dir = sd.rotate_vector(sd.get_orient_q(state), (0, 0, 1))
        try:
            freestream_right_dir = sd.normalize_or_err(np.cross(belly_dir, v))
            lift_dir = sd.normalize_or_err(np.cross(freestream_right_dir, v))
        except ValueError:
            return sd.make_wrench()
        force = 0.5 * rho * np.dot(v, v) * self.lift_coeff * self.wing_area * lift_dir
        return sd.make_wrench(lin=force)
