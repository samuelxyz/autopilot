import numpy as np
import quaternion as quat

import state_def as sd
import actor

class Body:
    '''Representation of a spherical celestial body such as the Earth or Moon'''
    def __init__(self, name, body_state, radius, GM):
        self.name = name
        self.state = np.asarray(body_state)
        self.radius = radius
        self.gravity = actor.Gravity(self.state[sd.POS], GM)

    def update(self, dt):
        # future features may include stuff like solar system simulation to move bodies
        sd.rotate_state(self.state, quat.from_rotation_vector(dt * self.state[sd.ANGVEL]))
        self.state[sd.POS] += dt * self.state[sd.VEL]
        # self.gravity.pos = self.state[sd.POS] probably not needed since it's a view

    def check_collision(self, ship_state):
        return np.linalg.norm(ship_state[sd.POS] - self.state[sd.POS]) <= self.radius
    
    def get_NED_frame(self, ship_pos):
        '''Get the rotation matrix representing the North, East, Down frame for the given ship state.
        First column is the unit vector pointing north from the ship's position, etc.
        The rotation matrix transforms a vector from the NED frame to the system frame (probably ICRF).
        
        Returns nans if the NED frame is indeterminate (ship is on this body's polar axis).
        '''
        try:
            u_down = sd.normalize_or_err(self.state[sd.POS] - ship_pos)
            u_east = sd.normalize_or_err(np.linalg.cross(u_down, self.state[sd.ANGVEL]))
            u_north = np.linalg.cross(u_east, u_down)
            NED = np.column_stack((u_north, u_east, u_down))
            return NED
        except ValueError:
            return np.full((3,3), np.nan)
    
    def get_euler_angles_NED(self, ship_state):
        '''Get Euler or Tait-Bryan angles (radians) describing the ship's attitude 
        using heading, elevation, and bank according to aircraft conventions:
        * Reference frame x, y, z vectors are local North, East, Down directions respectively.

        Returns nans if the NED frame is indeterminate (ship is on this body's polar axis).
        In gimbal lock, the indeterminate angles will be returned as nan.
        '''
        NED = self.get_NED_frame(ship_state[sd.POS])
        if np.any(np.isnan(NED)):
            return np.full(3, np.nan)
        return sd.heading_elevation_bank_FRB(sd.get_orient(ship_state), NED)

class Environment:
    '''Context to track simulation parameters such as colliding with celestial bodies'''
    def __init__(self, bodies:list[Body]):
        self.bodies = bodies

    def get_gravity_actors(self):
        return [body.gravity for body in self.bodies]
    
    def check_for_body_collisions(self, ship_state):
        '''If the ship state has overlapped a body, return the body, else None'''
        for body in self.bodies:
            if body.check_collision(ship_state):
                return body
        return None

def make_environment_ECI():
    '''Contains Earth centered at the origin. 
    Frame is oriented with Earth's north pole along +z, similar to J2000/EME2000/ICRF frames.
    '''
    return Environment([
        Body('Earth', sd.make_state(angvel=(0, 0, 2*np.pi/86164.0905)), 6371e3, actor.GM_Earth)
    ])

if __name__ == '__main__':
    env = make_environment_ECI()
    earth = env.bodies[0]
    ship_state = sd.state_from_orbit_properties(actor.GM_Earth, period=1e5, eccentricity=0, true_anomaly=-0.1)
    print(ship_state)
    print(earth.get_euler_angles_NED(ship_state)) # 90deg, not quite 90deg, 90deg