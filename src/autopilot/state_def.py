# state_def.py
# Contains various type definitions and functions related to state vectors

import numpy as np
import quaternion as quat

# old stuff
# state = np.dtype([
#     ('pos', 'f4', (3,)),
#     ('orient', np.quaternion),
#     ('vel', 'f4', (3,)),
#     ('angvel', 'f4', (3,)),
#     ])

# qdot = np.dtype([
#     ('lin', 'f4', (3,)),
#     ('ang', 'f4', (3,)),
# ])

# State vector elements
#                             | vels ---------------- |
#   pos  ---- | orient ------ | vel ---- | angvel --- |
#   0   1   2   3   4   5   6   7   8   9   10  11  12
STATE_N = 13
NxN = (STATE_N, STATE_N)
POS    = slice(0, 3) # any vector v + this will convert v from local into world frame
ORIENT = slice(3, 7) # applying to any vector v will convert v from local into world frame
VEL    = slice(7, 10)
ANGVEL = slice(10, 13)
VELS   = slice(7, 13)

def make_zero_state():
    state = np.zeros(STATE_N)
    state[ORIENT.start] = 1
    return state

def get_orient(state: np.array) -> np.array:
    '''Extract a view from the state or array of states, containing orientation 
    or array of orientations, as quaternions'''
    return quat.from_float_array(state[..., ORIENT])

def rotate_state(state: np.array, rotation: np.quaternion):
    '''Update only the orientation of the state or array of states. Other parts of 
    the generalized coordinates do not get coordinate transforms applied or anything'''
    # multiplying quat on the left == compose rotations
    state[..., ORIENT] = quat.as_float_array(rotation * get_orient(state))

def normalize_quat_part(state: np.array):
    state[ORIENT] = quat.as_float_array(np.normalized(get_orient(state)))

# Not sure where this would be better situated but i'll put it here for now
def as_4x4_matrix(quat):
    '''Convert quat into a 4x4 matrix Q, such that Q * as_float_array(other_quat) == quat * other_quat'''
    return np.array([
        [quat.w, -quat.x, -quat.y, -quat.z,],
        [quat.x,  quat.w, -quat.z,  quat.y,],
        [quat.y,  quat.z,  quat.w, -quat.x,],
        [quat.z, -quat.y,  quat.x,  quat.w,],
    ])

# Action / qdot / ddt(state) elements
#   lin ----- | ang ----- |
#   0   1   2   3   4   5
QDOT_N = 6
LIN = slice(0, 3)
ANG = slice(3, 6)

def change_to_frame(frame: np.array, other: np.array):
    '''Returns a state vector describing `other` from the point of view of `frame`.
    Parameters and return values are all state vectors.
    '''
    # Probably has crazy bugs. hard to find reference material about transforming velocities 
    # and angular velocities between two independently moving, rotating, and oriented frames in 3D

    result = other.copy()
    rotation_into_frame_q = get_orient(frame).conj()
    rotation_into_frame_m = quat.as_rotation_matrix(rotation_into_frame_q)
    result[POS] = rotation_into_frame_m @ (other[POS] - frame[POS])
    rotate_state(result, rotation_into_frame_q) # affects orientation quat only
    result[VEL] = rotation_into_frame_m @ (other[VEL] - frame[VEL]) - np.cross(frame[ANGVEL], result[POS])
    result[ANGVEL] = rotation_into_frame_m @ (other[ANGVEL] - frame[ANGVEL])
    return result

def RK4_step(state, dt, accel_calculator):
    '''Propagate the given state given an acceleration function.
    
    Parameters:
        state: State at the start of timestep (will not be modified in-place)
        dt: Length of timestep
        accel_calculator: function matching accel_calculator(some_state) -> [linear and angular acceleration]
    Returns:
        A new state vector at end of timestep
    '''
    # RK4 ish
    # Call old conditions a0, v0, q0
    # Step 1 with dt/2, a0 -> v[0 -> 1] -> q[0 -> 1] -> a1
    accel0 = accel_calculator(state)
    state1 = state.copy()
    state1[VELS] += accel0*dt/2
    state1[POS] += state1[VEL]*dt/2
    rotate_state(state1, quat.from_rotation_vector(state1[ANGVEL]*dt/2))
    accel1 = accel_calculator(state1)
    # Step 2 with dt/2, a1 -> v[0 -> 2] -> q[0 -> 2] -> a2 
    state2 = state.copy()
    state2[VELS] += accel1*dt/2
    state2[POS] += state2[VEL]*dt/2
    rotate_state(state2, quat.from_rotation_vector(state2[ANGVEL]*dt/2))
    accel2 = accel_calculator(state2)
    # Step 3 with dt, a2 -> v[0 -> 3] -> q[0 -> 3] -> a3
    state3 = state.copy()
    state3[VELS] += accel2*dt/2
    state3[POS] += state3[VEL]*dt
    rotate_state(state3, quat.from_rotation_vector(state3[ANGVEL]*dt/2))
    accel3 = accel_calculator(state3)
    # Step 4 with dt, fractional weights

    result = state.copy()
    result[POS] += dt/6 * (state[VEL] + 2*state1[VEL] + 2*state2[VEL] + state3[VEL])
    rotate_state(result, quat.from_rotation_vector(dt/6 * (result[ANGVEL] + 2*state1[ANGVEL] + 2*state2[ANGVEL] + state3[ANGVEL])))
    result[VELS] += dt/6 * (accel0 + 2*accel1 + 2*accel2 + accel3)
    
    # normalize orientation quaternion
    # state[ORIENT] /= np.linalg.norm(state[ORIENT])
    normalize_quat_part(result)
    return result