import numpy as np
import state_def as sd

def make_state_transition_matrix(dt, state):
    '''13x13 matrix (according to state_def.STATE_N) that when multiplied 
    by a state x predicts the next state xprime. Intended for Extended Kalman Filter'''
    a, b, c = state[sd.ANGVEL] * dt/2

    return np.asarray([
        # x  y  z     qr  qx  qy  qz    vx vy vz    wx wy wz
        [ 1, 0, 0,    0,  0,  0,  0,    dt, 0, 0,   0, 0, 0 ], # x' = x + v_x*t
        [ 0, 1, 0,    0,  0,  0,  0,    0, dt, 0,   0, 0, 0 ],
        [ 0, 0, 1,    0,  0,  0,  0,    0, 0, dt,   0, 0, 0 ],

        [ 0, 0, 0,    1, -a, -b, -c,    0, 0, 0,    0, 0, 0 ], # q_r' = q_r + t/2 (        - w_x*q_x - w_y*q_y - w_z*q_z)
        [ 0, 0, 0,    a,  1, -c,  b,    0, 0, 0,    0, 0, 0 ], # q_x' = q_x + t/2 (w_x*q_r           - w_z*q_y + w_y*q_z)
        [ 0, 0, 0,    b,  c,  1, -a,    0, 0, 0,    0, 0, 0 ], # q_y' = q_y + t/2 (w_y*q_r + w_z*q_x           - w_x*q_z)
        [ 0, 0, 0,    c, -b,  a,  1,    0, 0, 0,    0, 0, 0 ], # q_y' = q_z + t/2 (w_z*q_r - w_y*q_x + w_x*q_z          )

        [ 0, 0, 0,    0,  0,  0,  0,    1, 0, 0,    0, 0, 0 ], # vx' = vx
        [ 0, 0, 0,    0,  0,  0,  0,    0, 1, 0,    0, 0, 0 ],
        [ 0, 0, 0,    0,  0,  0,  0,    0, 0, 1,    0, 0, 0 ],

        [ 0, 0, 0,    0,  0,  0,  0,    0, 0, 0,    1, 0, 0 ], # wx' = wx
        [ 0, 0, 0,    0,  0,  0,  0,    0, 0, 0,    0, 1, 0 ],
        [ 0, 0, 0,    0,  0,  0,  0,    0, 0, 0,    0, 0, 1 ],
    ])

def make_orient_transition_matrix(dt, omega):
    '''4x4 matrix intended for quaternion orientation. omega should be vec3'''
    a, b, c = omega * dt/2
    return np.asarray([
        [1, -a, -b, -c], # q_r' = q_r + t/2 (        - w_x*q_x - w_y*q_y - w_z*q_z)
        [a,  1, -c,  b], # q_x' = q_x + t/2 (w_x*q_r           - w_z*q_y + w_y*q_z)
        [b,  c,  1, -a], # q_y' = q_y + t/2 (w_y*q_r + w_z*q_x           - w_x*q_z)
        [c, -b,  a,  1], # q_y' = q_z + t/2 (w_z*q_r - w_y*q_x + w_x*q_z          )
    ])

def make_translational_transition_matrix(t):
    '''6x6 matrix intended for x, y, z, vx, vy, vz'''
    result = np.eye(6)
    result[0:3, 3:6] = t * np.eye(3)
    return result

class KalmanFilter:
    '''Class implementing a simple Kalman filter.
    
    Attributes:
        x: estimated state vector (Nx1), most recent
        P: estimated covariance of state vector (NxN), most recent
        A: State transition matrix (NxN) such that next x is approximately Ax
        H: matrix representing the measurement forward model (MxN) such that z = Hx
        Q: state transition noise variance (diagonal, NxN)
        R: measurement noise variance (diagonal, MxM)
        xp: a priori predicted state vector, most recent (less accurate than x but at the same timestep)
        Pp: a priori predicted covariance of the state vector, most recent (less accurate than P but at the same timestep)
        K: Kalman gain, most recent

    Methods:
        __init__(x, P, A, Q, H, R): Constructor
        step(A, Q): Update estimate using system model A (NxN)
        process_observation(z, H, R): Update estimate using measurement vector z (Mx1) and forward model H (MxN)
    '''

    def __init__(self, x, P, A=None, Q = None, H = None, R = None):
        '''Parameters:
            x: state vector (Nx1), initial estimate
            P: covariance of state vector (NxN), initial estimate
            A: State transition matrix (NxN) such that next x is approximately Ax
            Q: state transition noise (diagonal, NxN)
            H: forward model such that z = Hx (MxN)
            R: measurement noise (diagonal, MxM)
        '''
        self.x = np.asarray(x)
        self.P = np.asarray(P)
        assert np.ndim(self.x) == 1
        assert np.shape(self.P) == (np.size(self.x), np.size(self.x))

        # Uninitialized values may be useful as members for history module to point to
        self.A = np.asarray(A) if A is not None else np.full((self.x.size, self.x.size), np.nan)
        self.Q = np.asarray(Q) if Q is not None else np.full_like(self.A, np.nan)
        self.H = np.asarray(H) if H is not None else np.nan
        self.R = np.asarray(R) if R is not None else np.nan
        assert np.shape(self.A) == np.shape(self.P)
        assert np.shape(self.Q) == np.shape(self.A)

        self.xp = np.full_like(self.x, np.nan)
        self.Pp = np.full_like(self.P, np.nan)
        self.K = np.full_like(np.transpose(self.H), np.nan)

    def step(self, A: np.ndarray = None, Q: np.ndarray = None):
        '''Update the filter by predicting the next state vector (advance one timestep).
        
        Parameters:
            A: State transition matrix (NxN) such that next x is approximately Ax
            Q: state transition noise variance (diagonal, NxN)
            Omit either one to use previous (member) values
        
        Returns: 
            xp: New prior prediction of state vector (Nx1)
            Pp: New prior predicted covariance of state vector (NxN)
        
        Side-effects:
            Stores this step's priori estimates of x and P as self.xp and self.Pp
            Stores A and Q if given
        '''
        if A is not None:
            self.A = np.asarray(A)
        if Q is not None:
            self.Q = np.asarray(Q)
        self.xp = A @ self.x
        self.Pp = A @ self.P @ A.T + self.Q
        return self.xp, self.Pp

    def process_observation(self, z: np.ndarray, H: np.ndarray = None, R: np.ndarray = None):
        '''Update the filter by processing a new observation vector z. 
        
        Parameters:
            z: Observation vector (Mx1) to be processed
            H: forward model such that z = Hx (MxN)
            R: measurement noise (diagonal, MxM)
            Omit H or R to use previous (member) values
        
        Returns: 
            x: New prediction of state vector (Nx1)
            P: New predicted covariance of state vector (NxN)
        
        Side-effects:
            Stores new x and P as self.x and self.P (same as returned values)
            Stores this step's Kalman gain as self.K
            Stores H and R if given
        '''
        if H is not None:
            self.H = np.asarray(H)
        if R is not None:
            self.R = np.asarray(R)
        self.K = self.Pp @ self.H.T @ np.linalg.inv(self.H @ self.Pp @ self.H.T + self.R)
        self.x = self.xp + self.K @ (z - self.H @ self.xp)
        self.P = self.Pp - self.K @ self.H @ self.Pp

        return self.x, self.P
    
class ExtendedKalmanFilter(KalmanFilter):
    '''Class implementing an Extended Kalman filter.
    
    Attributes:
        x: estimated state vector (Nx1), most recent
        P: estimated covariance of state vector (NxN), most recent
        Q: state transition noise variance (diagonal, NxN)
        Hfunc: function representing the measurement forward model such that z = H(x)
        H: linearized measurement forward model (MxN) such that z = Hx
        R: measurement noise variance (diagonal, MxM)
        xp: a priori predicted state vector, most recent (less accurate than x but at the same timestep)
        Pp: a priori predicted covariance of the state vector, most recent (less accurate than P but at the same timestep)
        K: Kalman gain, most recent

    Methods:
        __init__(self, H, Q, R, x, P): Constructor
        step(self, xp, A, Q): Update estimate using forward model A (NxN)
        process_observation(self, z, H): Update estimate using measurement vector z (Mx1) and forward model H (MxN)
        '''
    
    def __init__(self, x, P, A=None, Q=None, H=None, R=None):
        super().__init__(x, P, A, Q, H, R)
        self.zp = np.nan

    def step(self, xp, A=None, Q=None):
        '''Update the filter by predicting the next state vector (advance one timestep).
        
        Parameters:
            xp: New prior prediction of state vector (Nx1) - Do it yourself
            A: Linearized state transition matrix (NxN) such that next x is approximately Ax
            Q: state transition noise variance (diagonal, NxN)
            Omit A or Q to use previous (member) values
        
        Returns: 
            Pp: New prior predicted covariance of state vector (NxN)
        
        Side-effects:
            Stores this step's priori estimates of x and P as self.xp and self.Pp
            Stores A and Q if given
        '''
        if A is not None:
            self.A = np.asarray(A)
        if Q is not None:
            self.Q = np.asarray(Q)
        self.xp = xp
        self.Pp = A @ self.P @ A.T + self.Q

    def process_observation(self, z, zp, H = None, R = None):
        '''Update the filter by processing a new observation vector z. 
        
        Parameters:
            z: Observation vector (Mx1) to be processed
            zp: Predicted z based on xp (Mx1) - Apply the forward model yourself
            H: Linearized forward model such that z = Hx (MxN)
            R: measurement noise (diagonal, MxM)
            Omit H or R to use previous (member) values
        
        Returns: 
            x: New prediction of state vector (Nx1)
            P: New predicted covariance of state vector (NxN)
        
        Side-effects:
            Stores new x and P as self.x and self.P (same as returned values)
            Stores this step's Kalman gain as self.K
            Stores H and R if given
        '''
        if H is not None:
            self.H = np.asarray(H)
        if R is not None:
            self.R = np.asarray(R)

        y = z - zp # innovation
        S = self.H @ self.Pp @ self.H.T + self.R # cov of innovation
        self.K = self.Pp @ self.H.T @ np.linalg.inv(S)
        self.x = self.xp + self.K @ y
        self.P = self.Pp - self.K @ self.H @ self.Pp

        return self.x, self.P

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sim_tools
    def kalman_test():
        num_steps = 100
        state_dims = 2
        t_max = 10
        time = np.linspace(0, t_max, num_steps)
        true_x_history = np.sin(np.linspace(0, 2*t_max, num_steps*state_dims).reshape(num_steps, state_dims, order='F'))
        # H = np.array([[1, 0], [1, 1]])
        H = np.eye(2)
        Q = np.diag([t_max/num_steps, t_max/num_steps])**2 # covariance
        noise = [0.3, 0.05] # standard deviation
        R = np.diag(noise)**2 # covariance
        x_guess = [0, 0]
        P_guess = np.diag([0.2**2, 1])

        rng = np.random.default_rng(1234)
        ideal_observations = np.matvec(H, true_x_history)
        noisy_observations = ideal_observations + rng.multivariate_normal([0, 0], R, num_steps)

        kalman_filter = KalmanFilter(x_guess, P_guess, Q, H, R)
        hist = sim_tools.History(num_steps, {
            'x': lambda: kalman_filter.x,
            'P': lambda: kalman_filter.P,
            'K': lambda: kalman_filter.K
        })
        A = np.eye(2)
        for i in range(num_steps):
            kalman_filter.step(A)
            kalman_filter.process_observation(noisy_observations[i])
            hist.save_timestep(i)

        fig, axs = plt.subplots(3, sharex=True)
        for i, ax in enumerate(axs[:2]):
            ax.plot(time, true_x_history[..., i], label='true')
            ax.plot(time, noisy_observations[..., i], label='observed')
            ax.plot(time, hist['x'][..., i], label='filtered')
            ax.fill_between(time, 
                            hist['x'][..., i] - 3*hist['P'][..., i, i], 
                            hist['x'][..., i] + 3*hist['P'][..., i, i],
                            alpha=0.2, color='C2')
            ax.set_ylabel(f'{("x", "y")[i]} (noise={noise[i]})')
            ax.grid()
            ax.legend()
        axs[2].plot(time, np.linalg.det(hist['K']))
        axs[2].set_ylabel('det(K)')
        axs[2].grid()
        axs[-1].set_xlabel('Time (s)')
        plt.show()
    kalman_test()
