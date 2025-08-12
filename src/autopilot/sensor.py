# sensor.py
# Models sensors of various types used for ship state determination
# Also contains a sensor integration system using an Extended Kalman Filter

import numpy as np
import quaternion as quat
import kalman
import state_def as sd

class Sensor:
    def __init__(self, dim, constant_noise, randseed):
        '''Base class for an arbitrary sensor ready to hook into an Extended Kalman Filter.
        
        Parameters:
            dim: Number of dimensions of the sensor output
            constant_noise: Standard deviation of noise in sensor output. Can be scalar, vector, or matrix
            randseed: Seed for noise generator, used for reproducibility
        '''
        self.dim = dim
        self.reading = np.full(dim, np.nan) # The last reading
        self.rng = np.random.default_rng(randseed)
        match np.ndim(constant_noise):
            case 0: self.cov_matrix = np.eye(dim) * constant_noise**2
            case 1: self.cov_matrix = np.diag(constant_noise**2)
            case 2: self.cov_matrix = constant_noise**2
            case _: self.cov_matrix = np.nan
    
    def is_available(self, true_state, time):
        '''Whether this sensor can currently produce an observation.'''
        return True
    
    def new_reading(self, true_state):
        '''Simulates an actual observation. Return the noisy observation.'''
        self.reading = np.zeros(self.dim)
        return self.reading
    
    def cov(self, state):
        '''Estimate error covariance of the sensor reading at the given state. For use with Kalman filters.'''
        return self.cov_matrix
    
    def predict_reading(self, state):
        '''Return zp, no noise. For use with Kalman filters.'''
        return np.zeros(self.dim)

    def linearized_model(self, state):
        '''Return H for Kalman Filter.'''
        return np.zeros((self.dim, sd.STATE_N))
    
class IntermittentSensor(Sensor):
    '''Base class for a sensor that can only sense at specified intervals.'''
    def __init__(self, dim, constant_noise, sensing_interval, randseed):
        super().__init__(dim, constant_noise, randseed)
        self.sensing_interval = sensing_interval
        self.last_sense_time = 0

    def is_available(self, true_state, time):
        if self.sensing_interval == 0:
            return True
        if time - self.last_sense_time > self.sensing_interval:
            self.last_sense_time = (time // self.sensing_interval) * self.sensing_interval
            return True
        else:
            return False
    
class Gyroscope(Sensor):
    '''A sensor that directly reads angular velocity in the local frame.'''
    # SMAD: biases typically 0.003 - 1 deg/hr (1.5e-8 - 5e-6 rad/s), drift in bias varies widely
    # Mass: <0.1 - 15 kg
    # Power: <1 to 200 W

    def __init__(self, constant_noise, randseed=None):
        super().__init__(3, constant_noise, randseed)

    def new_reading(self, true_state):
        ideal_reading = quat.rotate_vectors(sd.get_orient(true_state).conj(), true_state[sd.ANGVEL])
        reading = self.rng.multivariate_normal(ideal_reading, self.cov_matrix)
        self.reading = reading
        return reading
    
    def predict_reading(self, state):
        return quat.rotate_vectors(sd.get_orient(state).conj(), state[sd.ANGVEL])
    
    def linearized_model(self, state):
        # h(x) = matrix(orientation.conj) * angvel
        # therefore, H = dh/dx = matrix(orientation.conj) appropriately padded with zeros
        H = np.zeros((3, sd.STATE_N))
        H[:, sd.ANGVEL] = quat.as_rotation_matrix(sd.get_orient(state).conj())
        return H
    
class StarTracker(IntermittentSensor):
    '''An accurate orientation sensor that is only available at certain time intervals.'''
    # SMAD: 1 arcsec - 1 arcmin (5e-6 - 3e-4 rad)
    # Mass: 2-5 kg
    # Power: 5-20 W

    # For Kalman filter
    H = np.zeros((4, sd.STATE_N))
    H[:, sd.ORIENT] = np.eye(4)

    def __init__(self, constant_noise, sensing_interval=0., randseed=None):
        '''Special case here, the noise represents a random rotation vector (3) 
        instead of quat components (4). Still accepts scalar, vector, or matrix.
        '''
        super().__init__(4, constant_noise, sensing_interval, randseed)
        self.cov_matrix = self.cov_matrix[:3, :3]
    
    def new_reading(self, true_state):
        # Small rotation vector
        perturbation = self.rng.multivariate_normal([0, 0, 0], self.cov_matrix)
        perturbation_q = quat.from_rotation_vector(perturbation)
        self.reading = perturbation_q * sd.get_orient(true_state)
        return quat.as_float_array(self.reading)
    
    def predict_reading(self, state):
        return state[sd.ORIENT]
    
    def cov(self, state):
        # How to convert vector covariance (small rotation vector) to quat covariance?
        # The observation function adds a small rotation vector to the current state rotation:
        #   First, convert small rotation vector w to a quaternion qpert = Ew + (1, 0, 0, 0)
        #     (Note the quaternion is [cos(theta/2), w*sin[theta/2], then small angle approximation 
        #      gives [1, w/2]. So E = a row of 0, then 3 rows of I/2)
        #   Then, apply qpert to the actual rotation z = qpert * q. Represent this as a 4x4 matrix Qpert * q
        #   Well since it's noise i think the order is reversible so just do q * Qpert instead
        #   Convert q to a 4x4 matrix using Q = sd.as_4x4_matrix(q)
        # So z = Q(Ew + [1, 0, 0, 0]) is a linearized version of the observation function
        E = np.zeros((4, 3))
        E[1:, :] = 0.5 * np.eye(3)
        Q = sd.as_4x4_matrix(sd.get_orient(state))
        QE = Q @ E
        return QE @ self.cov_matrix @ QE.T
        #  I hope this whole thing was at least remotely correct
    
    def linearized_model(self, state):
        return StarTracker.H

class HorizonSensor(Sensor):
    # SMAD: Scanning type 0.1 - 0.25 deg (2e-3 - 4e-3 rad) for LEO, down to 0.03 deg (5e-4 rad) in non LEO
    # Mass: 2-5 kg
    # Power: 5-10 W
    
    # Fixed type for rotating spacecraft: <0.1 - 0.25 deg (2e-3 - 4e-3 rad)
    # Mass: 0.5 - 3.5 kg
    # Power: 0.3 - 5 W

    pass # TODO

class SunSensor(Sensor):
    # SMAD: Accuracy 0.005 - 3 deg (1e-4 - 5e-2 rad)
    # Mass: 0.1 - 2 kg
    # Power: 0-3 W
    pass # TODO

class DistanceSignal(IntermittentSensor):
    '''A simulated sensor comprising a distance reading from a single GPS-like satellite whose position is known. 
    Sensor output is the distance to this satellite.'''
    
    def __init__(self, satellite_state, satellite_actors, constant_noise, sensing_interval, randseed=None):
        super().__init__(1, constant_noise, sensing_interval, randseed)
        self.satellite_state = satellite_state
        self.satellite_actors = satellite_actors

    # TODO: add occultation checks for is_available()

    def update_simulation(self, dt):
        def calc_accel(state):
            result = np.zeros((sd.QDOT_N,))
            for actor in self.satellite_actors:
                result += actor.get_accel(state, None, None)
            return result
        self.satellite_state = sd.RK4_step(self.satellite_state, dt, calc_accel)

    def new_reading(self, true_state):
        distance = np.linalg.vector_norm(true_state[sd.POS] - self.satellite_state[sd.POS])
        assert self.cov_matrix.size == 1
        self.reading = (distance + self.rng.normal(0, np.sqrt(self.cov_matrix))).reshape(1)
        return self.reading
    
    def predict_reading(self, state):
        return np.linalg.vector_norm(state[sd.POS] - self.satellite_state[sd.POS])
    
    def linearized_model(self, state):
        H = np.zeros(sd.STATE_N)
        x_minus_x0 = state[sd.POS] - self.satellite_state[sd.POS]
        H[sd.POS] = x_minus_x0 / np.linalg.vector_norm(x_minus_x0)
        return H[np.newaxis, :]

class PositionSensor(IntermittentSensor):
    '''An idealized sensor that simply outputs the ship's absolute position. 
    Can be thought of as the output of a more sophisticated GPS receiver subsystem.'''

class Radar(Sensor):
    pass # TODO

class StateDeterminationSystem:
    '''Sensor fusion system based around the Extended Kalman Filter.'''

    def __init__(self, sensors: list[Sensor], x, P, Q):
        self.sensors = sensors
        self.Q = Q
        self.EKF = kalman.ExtendedKalmanFilter(x, P, Q)
        self.is_up_to_date = True # Whether self.EKF.x is better to use than self.EKF.xp

    def predict_step(self, dt, accel_predictor):
        '''Run the prediction step of the EKF, propagating the estimated state by one timestep.
        Updates EKF.xp and EKF.Pp, but not EKF.x or EKF.P'''

        xp = sd.RK4_step(self.EKF.x, dt, accel_predictor)

        A = kalman.make_state_transition_matrix(dt, self.EKF.x)
        self.EKF.predict_step(xp, A, self.Q)        
        sd.normalize_quat_part(self.EKF.x)
        self.is_up_to_date = False

    def sense(self, true_state, timestamp):
        '''Take a reading from all available sensors, updating the EKF.'''
        active_sensors = [s for s in self.sensors if s.is_available(true_state, timestamp)]
        H  = np.vstack(tuple(np.atleast_1d(s.linearized_model(self.EKF.xp)) for s in active_sensors)) 
        zp = np.concat(tuple(np.atleast_1d(s.predict_reading(self.EKF.xp)) for s in active_sensors)).T
        z  = np.concat(tuple(np.atleast_1d(s.new_reading(true_state)) for s in active_sensors)).T
        
        # assemble R
        R = np.eye(z.size)
        i = 0
        for s in active_sensors:
            R_sensor = s.cov(self.EKF.xp)
            inds = slice(i, i+R_sensor.shape[0])
            R[inds, inds] = R_sensor
            i += R_sensor.shape[0]

        self.EKF.process_observation(z, zp, H, R)
        sd.normalize_quat_part(self.EKF.x)
        self.is_up_to_date = True

    def get_current_state_estimate(self):
        if self.is_up_to_date:
            return self.EKF.x, self.EKF.P
        else:
            return self.EKF.xp, self.EKF.Pp