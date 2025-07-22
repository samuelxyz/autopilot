# Ship
# Contains a 6-DOF rigid body simulation that will be controlled by various actuators and controllers
# RK4 time-stepping

import numpy as np
import quaternion as quat
import actor, controller
import state_def as sd

class Ship:

    def __init__(self, state, mass, inertia_matrix, controller:controller.Controller, actors: list[actor.Actor] = []):
        self.state = np.asarray(state) # current state, updated every timestep. q, qdot
        # self.state_history = ...
        self.mass = mass # something
        self.inertia_matrix = inertia_matrix # body frame
        self.controller = controller
        self.actors = actors
        self.setpoint = np.ones(3)

    def update(self, dt):
        self.controller.update(self.state, dt)
        self.do_motion_timestep(dt)

    def do_motion_timestep(self, dt):
        # RK4 ish
        # Call old conditions a0, v0, q0
        # Step 1 with dt/2, a0 -> v[0 -> 1] -> q[0 -> 1] -> a1
        accel0 = self.calc_accel(self.state)
        state1 = self.state.copy()
        state1[sd.VELS] += accel0*dt/2
        state1[sd.POS] += state1[sd.VEL]*dt/2
        sd.rotate(state1, quat.from_rotation_vector(state1[sd.ANGVEL]*dt/2))
        accel1 = self.calc_accel(state1)
        # Step 2 with dt/2, a1 -> v[0 -> 2] -> q[0 -> 2] -> a2 
        state2 = self.state.copy()
        state2[sd.VELS] += accel1*dt/2
        state2[sd.POS] += state2[sd.VEL]*dt/2
        sd.rotate(state2, quat.from_rotation_vector(state2[sd.ANGVEL]*dt/2))
        accel2 = self.calc_accel(state2)
        # Step 3 with dt, a2 -> v[0 -> 3] -> q[0 -> 3] -> a3
        state3 = self.state.copy()
        state3[sd.VELS] += accel2*dt/2
        state3[sd.POS] += state3[sd.VEL]*dt
        sd.rotate(state3, quat.from_rotation_vector(state3[sd.ANGVEL]*dt/2))
        accel3 = self.calc_accel(state3)
        # Step 4 with dt, fractional weights
        self.state[sd.POS] += dt/6 * (self.state[sd.VEL] + 2*state1[sd.VEL] + 2*state2[sd.VEL] + state3[sd.VEL])
        sd.rotate(self.state, quat.from_rotation_vector(dt/6 * (self.state[sd.ANGVEL] + 2*state1[sd.ANGVEL] + 2*state2[sd.ANGVEL] + state3[sd.ANGVEL])))
        self.state[sd.VELS] += dt/6 * (accel0 + 2*accel1 + 2*accel2 + accel3)
        
        # normalize orientation quaternion
        # self.state[sd.ORIENT] /= np.linalg.norm(self.state[sd.ORIENT])
        self.state[sd.ORIENT] = quat.as_float_array(np.normalized(sd.get_orient(self.state)))

        pass

    def calc_accel(self, state) -> np.ndarray:
        '''Calculates the linear and angular acceleration (total shape (6,)) from actors on the current state'''
        result = np.zeros((sd.QDOT_N,))
        # rotation_matrix = quat.as_rotation_matrix(sd.get_orient(self.state))
        # world_inertia_matrix = rotation_matrix @ self.inertia_matrix @ rotation_matrix.T
        for actor in self.actors:
            result += actor.get_accel(state, self.mass, self.inertia_matrix)
        return result