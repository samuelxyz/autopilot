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
        '''Estimate error covariance at state. For use with Kalman filters.'''
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
    pass # TODO

class SunSensor(Sensor):
    pass # TODO

class GPSReceiver(Sensor):
    pass # TODO

class Radar(Sensor):
    pass # TODO

class StateDeterminationSystem:
    def __init__(self, sensors: list[Sensor], x, P, Q):
        self.sensors = sensors
        self.Q = Q
        self.EKF = kalman.ExtendedKalmanFilter(x, P, Q)

    def update(self, true_state, start_time, dt, predicted_accel):
        active_sensors = [s for s in self.sensors if s.is_available(true_state, start_time)]

        # Predict
        xp, A = self._predict_next_state(dt, predicted_accel)
        self.EKF.step(xp, A, self.Q)

        # Update
        H  = np.vstack(tuple(s.linearized_model(self.EKF.xp) for s in active_sensors))
        zp = np.concat(tuple(s.predict_reading(self.EKF.xp) for s in active_sensors)).T
        z  = np.concat(tuple(s.new_reading(true_state) for s in active_sensors)).T
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

    def _predict_next_state(self, dt, predicted_accel):
        # Could use RK4 but ehh
        xp = self.EKF.x.copy()
        xp[sd.VELS] += predicted_accel * dt
        xp[sd.POS] += xp[sd.VEL] * dt
        sd.rotate_state(xp, quat.from_rotation_vector(xp[sd.ANGVEL] * dt))
        return xp, kalman.make_state_transition_matrix(dt, self.EKF.x)

    def get_current_state_estimate(self):
        return self.EKF.x, self.EKF.P