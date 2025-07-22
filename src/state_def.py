# util.py
# Contains various reference material, constants, and type definitions

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
POS    = slice(0, 3)
ORIENT = slice(3, 7)
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

def rotate(state: np.array, rotation: np.quaternion):
    '''Update only the orientation of the state or array of states. Other parts of 
    the generalized coordinates do not get coordinate transforms applied or anything'''
    # multiplying quat on the left == compose rotations
    state[..., ORIENT] = quat.as_float_array(rotation * get_orient(state))

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
    rotate(result, rotation_into_frame_q) # affects orientation quat only
    result[VEL] = rotation_into_frame_m @ (other[VEL] - frame[VEL]) - np.cross(frame[ANGVEL], result[POS])
    result[ANGVEL] = rotation_into_frame_m @ (other[ANGVEL] - frame[ANGVEL])
    return result