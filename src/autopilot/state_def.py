# state_def.py
# Contains various type definitions and functions related to state vectors

import numpy as np
import quaternion as quat

# State vector elements
#                             | vels ---------------- |
#   pos  ---- | orient ------ | vel ---- | angvel --- |
#   0   1   2   3   4   5   6   7   8   9   10  11  12
STATE_N = 13
NxN = (STATE_N, STATE_N)
POS = slice(0, 3)  # any vector v + this will convert v from local into world frame
ORIENT = slice(
    3, 7
)  # applying to any vector v will convert v from local into world frame
VEL = slice(7, 10)
ANGVEL = slice(10, 13)
VELS = slice(7, 13)

# Low-level functions for manipulating states


def make_state(pos=0.0, orient_q=None, vel=0.0, angvel=0.0):
    state = np.empty(STATE_N)
    state[POS] = pos
    state[ORIENT] = orient_q if orient_q is not None else (1, 0, 0, 0)
    state[VEL] = vel
    state[ANGVEL] = angvel
    return state


def get_orient_q(state: np.array) -> np.array:
    """Extract a view from the state or array of states, containing orientation
    or array of orientations, as quaternions"""
    return quat.from_float_array(state[..., ORIENT])


def rotate_state(state: np.array, rotation_q: np.quaternion):
    """Update only the orientation of the state or array of states. Other parts of
    the generalized coordinates do not get coordinate transforms applied or anything"""
    # multiplying quat on the left == compose rotations
    state[..., ORIENT] = quat.as_float_array(rotation_q * get_orient_q(state))


def normalize_quat_part(state: np.array):
    state[ORIENT] = quat.as_float_array(np.normalized(get_orient_q(state)))


# Not sure where this would be better situated but i'll put it here for now
def as_4x4_matrix(quat):
    """Convert quat into a 4x4 matrix Q, such that Q * as_float_array(other_quat) == quat * other_quat"""
    # fmt: off
    return np.array([
        [quat.w, -quat.x, -quat.y, -quat.z,],
        [quat.x,  quat.w, -quat.z,  quat.y,],
        [quat.y,  quat.z,  quat.w, -quat.x,],
        [quat.z, -quat.y,  quat.x,  quat.w,],
    ]) # fmt: on


def rotate_vector(rot_q, vec):
    """Apply quaternion to a single vector; should be faster than quat.rotate_vectors(q, v)."""
    qv = quat.from_vector_part(vec)
    rotated_qv = rot_q * qv * rot_q.conj()
    return quat.as_vector_part(rotated_qv)


# Wrench elements
# A wrench is a 6-element array formed by concatenating a linear and angular something, often force and torque
#   lin ----- | ang ----- |
#   0   1   2   3   4   5
WRENCH_N = 6
LIN = slice(0, 3)
ANG = slice(3, 6)


def make_wrench(lin=0.0, ang=0.0):
    wrench = np.empty(WRENCH_N)
    wrench[LIN] = lin
    wrench[ANG] = ang
    return wrench


# k is a 2xWRENCH_N array, first row is velocity, second row is acceleration.
# 6 columns represent LIN, ANG (3 entries each)
K_VEL = 0
K_ACC = 1

# High-level functions for manipulating states


def change_to_frame(frame: np.array, other: np.array):
    """Returns a state vector describing `other` from the point of view of `frame`.
    Parameters and return values are all state vectors.
    """
    # This was a fun exercise but I dont think I will use it much.
    # Probably has crazy bugs. hard to find reference material about transforming velocities
    # and angular velocities between two independently moving, rotating, and oriented frames in 3D

    result = other.copy()
    rotation_into_frame_q = get_orient_q(frame).conj()
    result[POS] = rotate_vector(rotation_into_frame_q, other[POS] - frame[POS])
    rotate_state(result, rotation_into_frame_q)  # affects orientation quat only
    result[VEL] = rotate_vector(
        rotation_into_frame_q, other[VEL] - frame[VEL]
    ) - np.cross(frame[ANGVEL], result[POS])
    result[ANGVEL] = rotate_vector(rotation_into_frame_q, other[ANGVEL] - frame[ANGVEL])
    return result


def RK4_step(state, dt, accel_calculator):
    """Propagate the given state given an acceleration function.

    Parameters:
        state: State at the start of timestep (will not be modified in-place)
        dt: Length of timestep
        accel_calculator: function matching accel_calculator(some_state) -> [linear and angular acceleration]
    Returns:
        A new state vector at end of timestep
    """
    # RK4 ish
    # Call old conditions a0, v0, q0
    # Step 1 with dt/2, a0 -> v[0 -> 1] -> q[0 -> 1] -> a1
    accel0 = accel_calculator(state)
    state1 = state.copy()
    state1[VELS] += accel0 * dt / 2
    state1[POS] += state1[VEL] * dt / 2
    rotate_state(state1, quat.from_rotation_vector(state1[ANGVEL] * dt / 2))
    accel1 = accel_calculator(state1)
    # Step 2 with dt/2, a1 -> v[0 -> 2] -> q[0 -> 2] -> a2
    state2 = state.copy()
    state2[VELS] += accel1 * dt / 2
    state2[POS] += state2[VEL] * dt / 2
    rotate_state(state2, quat.from_rotation_vector(state2[ANGVEL] * dt / 2))
    accel2 = accel_calculator(state2)
    # Step 3 with dt, a2 -> v[0 -> 3] -> q[0 -> 3] -> a3
    state3 = state.copy()
    state3[VELS] += accel2 * dt / 2
    state3[POS] += state3[VEL] * dt
    rotate_state(state3, quat.from_rotation_vector(state3[ANGVEL] * dt / 2))
    accel3 = accel_calculator(state3)
    # Step 4 with dt, fractional weights

    result = state.copy()
    result[POS] += (
        dt / 6 * (state[VEL] + 2 * state1[VEL] + 2 * state2[VEL] + state3[VEL])
    )
    rotate_state(
        result,
        quat.from_rotation_vector(
            dt
            / 6
            * (
                result[ANGVEL]
                + 2 * state1[ANGVEL]
                + 2 * state2[ANGVEL]
                + state3[ANGVEL]
            )
        ),
    )
    result[VELS] += dt / 6 * (accel0 + 2 * accel1 + 2 * accel2 + accel3)

    # normalize orientation quaternion
    # state[ORIENT] /= np.linalg.norm(state[ORIENT])
    normalize_quat_part(result)
    return result


def increment_state(state, state_increment):
    """Increment the given state by the given increment.

    Parameters:
        state: initial state, will not be modified
        state_increment: (2x6) array matching the form of k * timestep (see k in state_def).
            Or in other words, this is state2-state1 but with rotation vector instead of quaternion, and reshaped.

    """
    new_state = state.copy()
    # Note we use K_VEL here because state_increment is assumed to already be multiplied by time
    new_state[POS] += state_increment[K_VEL, LIN]
    rotate_state(new_state, quat.from_rotation_vector(state_increment[K_VEL, ANG]))
    # Note that by how this is set up, this equality is true
    new_state[VELS] = state_increment[K_ACC]

    return new_state


def RK4_newtoneuler_step(state, mass, inertia_matrix_body, dt, get_wrench_world):
    """Propagate the given rigid body state given a wrench function.

    Parameters:
        state: State at the start of timestep (will not be modified in-place)
        mass: Mass of rigid body
        inertia_matrix_body: Moment of inertia matrix of rigid body, body frame, about center of mass
        dt: Length of timestep
        get_wrench_world: function matching get_wrench_world(some_state) -> [(6,) vector of total applied force and moment]
    Returns:
        A new state vector at end of timestep
    """
    # RK4 ish
    # This entire function is written a bit verbosely (like not one-lining things) for easier debugging

    # game plan:
    # * calculate k1
    #   * wr1 = get_wrench_world(state)
    #   [enter scope of calc_statedot]
    #   * convert wr1 to k1 = (vels, accs)
    # * calculate k2
    #   * state1 = state "+" k1 * dt/2
    #   [leave scope of calc_statedot]
    #   * wr2 = get_wrench_world(state1)
    #   * convert wr2 to k2 = (vels, accs)
    # * calculate k3
    #   * state2 = state "+" k2 * dt/2
    #   * wr3 = get_wrench_world(state2)
    #   * convert wr3 to k3 = (vels, accs)
    # * calculate k4
    #   * state3 = state "+" k3 * dt
    #   * wr4 = get_wrench_world(state3)
    #   * convert wr4 to k4 = (vels, accs)
    # * combine
    #   * new_state = state "+"" (weighted sum of k's)

    def _calc_statedot(angvel_iminusone, wrench_i, tstep):
        k_i = np.zeros((2, WRENCH_N))
        k_i[K_ACC, LIN] = wrench_i[LIN] / mass
        k_i[K_VEL, LIN] = state[VEL] + k_i[K_ACC, LIN] * tstep
        # Do Euler's rotational equation of motion in the body frame
        # See https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        quat_to_world = quat.from_rotation_vector(
            angvel_iminusone * tstep
        ) * quat.get_orient_q(state)
        angvel_body = rotate_vector(quat_to_world.conj(), angvel_iminusone)
        moment_body = rotate_vector(quat_to_world.conj(), wrench_i[ANG])
        Ialpha_body = moment_body - np.cross(
            angvel_body, inertia_matrix_body @ angvel_body
        )
        alpha_body = np.linalg.solve(inertia_matrix_body, Ialpha_body)
        k_i[K_ACC, ANG] = rotate_vector(quat_to_world, alpha_body)
        k_i[K_VEL, ANG] = state[ANGVEL] + k_i[K_ACC, ANG] * tstep

        state_i = increment_state(state, k_i * tstep)

        return k_i, state_i

    wrench1 = get_wrench_world(state)
    k1, state1 = _calc_statedot(state[ANGVEL], wrench1, dt / 2)
    wrench2 = get_wrench_world(state1)
    k2, state2 = _calc_statedot(state1[ANGVEL], wrench2, dt / 2)
    wrench3 = get_wrench_world(state2)
    k3, state3 = _calc_statedot(state2[ANGVEL], wrench3, dt / 2)
    wrench4 = get_wrench_world(state3)
    k4, _ = _calc_statedot(state3[ANGVEL], wrench4, dt)

    k_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    new_state = increment_state(state, k_avg * dt)
    return new_state


def state_from_orbit_properties(
    GM,
    center=[0, 0, 0],
    semimajor_axis=None,
    period=None,
    eccentricity=None,
    periapsis=None,
    apoapsis=None,
    inclination=0.0,
    long_asc_node=0.0,
    arg_periapsis=0.0,
    true_anomaly=0.0,
    body_orient=None,
    body_angvel=0.0,
):
    """Generate a state vector from various orbital properties (consistent distance/time units, radians).

    Use exactly one of these combos:
    * Semimajor axis a and eccentricity e
    * Period T and eccentricity e
    * Periapsis q and apoapsis Q
    * Periapsis q and eccentricity e
    """
    # First unscramble all the optional params into a and e
    a = None  # semimajor axis
    e = None  # eccentricity
    if semimajor_axis is not None:
        a = semimajor_axis
        e = eccentricity
    elif period is not None:
        a = np.cbrt(GM * (period / (2 * np.pi)) ** 2)
        e = eccentricity
    elif periapsis is not None and apoapsis is not None:
        a = (periapsis + apoapsis) / 2
        e = (apoapsis - periapsis) / (apoapsis + periapsis)
    elif periapsis is not None and eccentricity is not None:
        e = eccentricity
        a = periapsis * (1 - e)
    assert a is not None and e is not None, (
        'Unsupported combination of orbit parameters given'
    )

    p = a * (1 - e**2)  # semi-latus rectum
    h = np.sqrt(GM * p)  # specific angular momentum scalar
    nu = true_anomaly

    # Source: https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html
    # In the orbit perifocal frame (x toward periapsis, z toward angular momentum)
    pos_w = h**2 / GM / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    vel_w = GM / h * np.array([-np.sin(nu), e + np.cos(nu), 0])

    # Rotate out of perifocal frame
    om = arg_periapsis
    Om = long_asc_node
    i = inclination
    # fmt: off
    R1 = np.array([[ np.cos(om), np.sin(om), 0],
                   [-np.sin(om), np.cos(om), 0],
                   [0, 0, 1]])
    R2 = np.array([[1, 0, 0,],
                   [0,  np.cos(i), np.sin(i)],
                   [0, -np.sin(i), np.cos(i)]])
    R3 = np.array([[ np.cos(Om), np.sin(Om), 0],
                   [-np.sin(Om), np.cos(Om), 0],
                   [0, 0, 1]])
    # fmt: on
    R = R1 @ R2 @ R3
    pos = pos_w @ R
    vel = vel_w @ R

    return make_state(pos + center, body_orient, vel, body_angvel)


class ReferenceFrameError(ValueError):
    """Could not construct reference frame"""

    pass


def normalize_or_err(vec):
    """Normalizes the given vector in-place to unit length. If it is zero, raise a ValueError instead."""
    vec = np.asarray(vec)
    if (norm := np.linalg.norm(vec)) < np.finfo(vec.dtype).eps:
        raise ValueError('Tried to normalize a zero vector')
    else:
        vec /= norm
        return vec


def normalize(vec):
    """Normalizes the given vector in-place to unit length. If it is zero, produces np.nan."""
    vec = np.asarray(vec)
    norm = np.linalg.norm(vec)
    vec /= norm
    return vec


def get_heading_elevation_bank_FRB(ship_orient_q, WRT_mat=np.eye(3)):
    """Get Euler or Tait-Bryan angles (radians) describing the ship's attitude
    using heading, elevation, and bank according to aircraft conventions:
    * Ship fixed frame x, y, z directions are front, right, belly (FRB) respectively
    * Right-handed intrinsic rotation associated with each axis is bank, elevation, heading respectively
    * Intrinsic rotations are applied in z-y'-x'' order

    Parameters:
        ship_orient_q: Orientation quaternion of the ship.
        WRT_mat: 3x3 matrix representing the frame that would ilne up with the ship at [0, 0, 0] Euler angles. For example, a North-East-Down frame.

    Returns:
        3-length numpy array containing the heading, elevation, and bank angles in radians.

    In gimbal lock, the indeterminate angles will be returned as nan.
    """

    FRB = quat.as_rotation_matrix(ship_orient_q)
    R = WRT_mat.T @ FRB  # FRB in WRT frame

    # Source: https://en.wikipedia.org/wiki/Euler_angles
    cos_elev = np.sqrt(1 - R[2, 0] ** 2)
    elevation = np.asin(-R[2, 0])
    # heading = np.asin(R[1, 0]/cos_elev)
    # bank = np.asin(R[2, 1]/cos_elev)

    # My implementation: from thinking real hard
    # Project the front-vector onto the north-east plane
    # Heading is the angle of the projection from north
    # heading = atan2(R_21, R_11) or atan2(R_12, R_11)
    heading = np.atan2(R[1, 0], R[0, 0])
    # at zero bank, ship's right-vector has no down-component
    # at 90deg bank, ship's right-vector has maximum down-component = cos(elevation)
    # at -90deg bank, ship's right-vector has minimum down-component = -cos(elevation) (is upward)
    # therefore, R_32 = down-component of right vector = cos(elevation)sin(bank)
    # Solve, sin(bank) = R_32 / cos(elevation)
    # Same as wikipedia solution
    # However, it is subject a bit to numerical errors. Let's fix that
    R32_over_cos_elev = R[2, 1] / cos_elev
    if 1.0 < abs(R32_over_cos_elev) and abs(R32_over_cos_elev) < 1.0 + 1e-9:
        R32_over_cos_elev = np.sign(R32_over_cos_elev)
    bank = np.asin(R32_over_cos_elev)

    return np.array([heading, elevation, bank])


def make_orientation_q_FRB(heading_elevation_bank, WRT_mat=np.eye(3)):
    """Make a quaternion that has the given heading, elevation, and bank angles
    relative to the given WRT frame. Uses aircraft conventions:
    * Ship fixed frame x, y, z directions are front, right, belly (FRB) respectively
    * Right-handed intrinsic rotation associated with each axis is bank, elevation, heading respectively
    * Intrinsic rotations are applied in z-y'-x'' order

    Parameters:
        heading_elevation_bank: length-3 sequence containing heading, elevation, and bank angle in radians.
        WRT_mat: 3x3 rotation matrix representing the frame that would line up with the ship
            at [0, 0, 0] Euler angles. For example, a North-East-Down frame.

    Returns:
        Unit quaternion representing the specified orientation.
    """
    s = np.sin(0.5 * np.asarray(heading_elevation_bank))
    c = np.cos(0.5 * np.asarray(heading_elevation_bank))

    # See https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    q_rel = quat.from_float_array(
        (
            c[2] * c[1] * c[0] + s[2] * s[1] * s[0],
            s[2] * c[1] * c[0] - c[2] * s[1] * s[0],
            c[2] * s[1] * c[0] + s[2] * c[1] * s[0],
            c[2] * c[1] * s[0] - s[2] * s[1] * c[0],
        )
    )

    result = quat.from_rotation_matrix(WRT_mat) * q_rel
    return np.normalized(result)


if __name__ == '__main__':
    # h = 19646.883e6
    # p = h^2/mu = 968389
    # a = p/(1-e^2) = 9559996
    test_state = state_from_orbit_properties(
        3.986e14,
        semimajor_axis=9559996,
        eccentricity=0.948,
        inclination=np.deg2rad(124.05),
        long_asc_node=np.deg2rad(190.62),
        arg_periapsis=np.deg2rad(303.09),
        true_anomaly=np.deg2rad(159.61),
    )

    def vector_error_prop(v, vref):
        delta = np.linalg.norm(v - vref)
        return delta / np.linalg.norm(vref)

    assert vector_error_prop(test_state[POS], np.array((1, 5, 7)) * 1e6) < 1e-2
    assert np.all(test_state[ORIENT] == np.array((1, 0, 0, 0)))
    assert vector_error_prop(test_state[VEL], np.array((3, 4, 5)) * 1e3) < 1e-2
    assert np.all(test_state[ANGVEL] == np.zeros(3))
    print(
        'Orbit state vector should be [1, 5, 7]e6, [1, 0, 0, 0], [3, 4, 5]e3, [0, 0, 0]\nActual:',
        test_state,
    )

    ship_setpoint_is_elevation_down_30deg = make_orientation_q_FRB(
        (0, -np.pi / 6, 0), np.eye(3)
    )
    ship_state_is_heading_east = make_orientation_q_FRB((np.pi / 2, 0, 0), np.eye(3))
    ship_setpoint_HEB = get_heading_elevation_bank_FRB(
        ship_setpoint_is_elevation_down_30deg, np.eye(3)
    )
    ship_state_HEB = get_heading_elevation_bank_FRB(
        ship_state_is_heading_east, np.eye(3)
    )
    assert np.linalg.norm(ship_setpoint_HEB - np.array((0, -np.pi / 6, 0))) < 1e-10
    assert np.linalg.norm(ship_state_HEB - np.array((np.pi / 2, 0, 0))) < 1e-10

    excess_orientation = (
        ship_setpoint_is_elevation_down_30deg.conj() * ship_state_is_heading_east
    )
    print(
        'Excess orientation HEB:',
        get_heading_elevation_bank_FRB(excess_orientation, np.eye(3)),
    )
