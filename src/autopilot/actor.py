# Actor
# Something that could apply forces on the ship: thrusters, gravity, aerodynamics, etc

import state_def as sd
import numpy as np

gravitational_constant_si = 6.6743e-11

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

    def get_accel(self, state, mass, inertia_matrix):
        r = state[sd.POS] - self.pos
        result = np.zeros((sd.QDOT_N,))
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