# Actor
# Something that could apply forces on the ship: thrusters, gravity, aerodynamics, etc

import state_def as sd
import numpy as np

gravitational_constant_si = 6.6743e-11
GM_Earth = 3.98600442e14
GM_Sun = 1.3271244002e20
GM_Moon = 4.902800118e12

def standard_atmospheric_density(altitude):
    '''Air density according to the standard reference atmosphere'''
    pass

class Actor:
    def __init__(self):
        pass

    def get_accel(self, state, mass, local_inertia_matrix):
        return np.zeros((sd.QDOT_N,))
    
class Gravity(Actor):
    
    def __init__(self, pos, GM):
        super().__init__()
        self.pos = np.asarray(pos)
        self.GM = GM

    def get_accel(self, state, mass, inertia_matrix, min_radius=1):
        '''min_radius is to avoid gravitational singularities'''
        r = state[sd.POS] - self.pos
        result = np.zeros((sd.QDOT_N,))
        if np.linalg.norm(r) < min_radius:
            return result # hardcoded to avoid bugginess
        result[sd.LIN] = -self.GM * r / np.linalg.norm(r)**3
        return result
    
class Drag_Equation(Actor):

    def __init__(self, drag_coeff, area, density, flow_field):
        super().__init__()
        self.drag_coeff = drag_coeff
        self.area = area # cross section
        self.density = density # callable(position)
        self.flow_field = flow_field # callable(position)

    def get_accel(self, state, mass, inertia_matrix):
        v = state[sd.VEL] - self.flow_field(state[sd.POS])
        rho = self.density(state[sd.POS])
        force = 0.5 * rho * np.dot(v, v) * self.drag_coeff * self.area

        result = np.zeros((sd.QDOT_N,))
        result[sd.LIN] = force/mass
        return result