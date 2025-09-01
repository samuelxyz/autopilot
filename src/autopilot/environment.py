import numpy as np
import quaternion as quat

import state_def as sd
import actor


class Body:
    """Representation of a planet-like body such as the Earth or Moon.
    Currently a pure sphere with Newtonian mechanics."""

    def __init__(self, name, body_state, radius, GM):
        self.name = name
        self.state = np.asarray(body_state)
        self.radius = radius
        self.gravity = actor.Gravity(self.state[sd.POS], GM)

    def update(self, dt):
        # future features may include stuff like N-body simulation to move bodies
        sd.rotate_state(
            self.state, quat.from_rotation_vector(dt * self.state[sd.ANGVEL])
        )
        self.state[sd.POS] += dt * self.state[sd.VEL]
        # self.gravity.pos = self.state[sd.POS] probably not needed since it's already a view

    def get_body_fixed_frame_q(self):
        """Get a quaternion representing the body-fixed frame relative to the system frame (probably ICRF).
        Applying this quaternion rotation to a vector transforms the vector
        from body-frame components to the system frame.

        Body frame is oriented with x pointing to 0,0 lat/long and z pointing to the north pole."""
        return sd.get_orient_q(self)

    def get_latlong_radians(self, ship_state):
        r_sys = ship_state[sd.POS] - self.state[sd.POS]
        r_body = quat.rotate_vectors(self.get_body_fixed_frame_q().conj(), r_sys)
        lat = np.asin(r_body[2] / np.linalg.norm(r_body))
        long = np.atan2(r_body[1], r_body[0])
        return np.asarray((lat, long))

    def get_altitude(self, ship_state):
        """Altitude of ship above body surface."""
        return np.linalg.norm(ship_state[sd.POS] - self.state[sd.POS]) - self.radius

    def check_collision(self, ship_state):
        return self.get_altitude(ship_state) <= 0

    def get_NED_frame_R(self, ship_pos):
        """Get a rotation matrix representing the North, East, Down frame for the given ship state.
        First column is the unit vector pointing north from the ship's position, etc.
        The rotation matrix transforms a vector from the NED frame to the system frame (probably ICRF).

        Raises sd.ReferenceFrameError if the NED frame is indeterminate (ship is on this body's polar axis).
        """

        try:
            u_down = sd.normalize_or_err(self.state[sd.POS] - ship_pos)
            u_east = sd.normalize_or_err(np.linalg.cross(u_down, self.state[sd.ANGVEL]))
        except ValueError as e:
            raise sd.ReferenceFrameError() from e
        u_north = np.linalg.cross(u_east, u_down)
        return np.column_stack((u_north, u_east, u_down))

    def get_euler_angles_NED(self, ship_state):
        """Get Euler or Tait-Bryan angles (radians) describing the ship's attitude
        using heading, elevation, and bank according to aircraft conventions:
        * Reference frame x, y, z vectors are local North, East, Down directions respectively.

        Raises sd.ReferenceFrameError if the NED frame is indeterminate (ship is on this body's polar axis).
        In gimbal lock, the indeterminate angles will be returned as nan.
        """
        NED = self.get_NED_frame_R(ship_state[sd.POS])
        return sd.heading_elevation_bank_FRB(sd.get_orient_q(ship_state), NED)

    def get_airspeed_vector(self, ship_state):
        """Returns the ship's velocity vector relative to this body in system reference frame (probably ICRF)."""
        r_vec = ship_state[sd.POS] - self.state[sd.POS]
        air_vel = self.state[sd.VEL] + np.cross(self.state[sd.ANGVEL], r_vec)
        return ship_state[sd.VEL] - air_vel

    def get_airspeed_vector_NED(self, ship_state):
        """Returns the ship's velocity vector relative to this body in the local North-East-Down frame.

        Raises sd.ReferenceFrameError if the NED frame is indeterminate (ship is on this body's polar axis).
        """
        NED = self.get_NED_frame_R(ship_state[sd.POS])
        return np.matvec(NED.T, self.get_airspeed_vector(ship_state))


# Simplified atmospheric property calculations
# See https://en.wikipedia.org/wiki/Barometric_formula
# See https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
# See https://en.wikipedia.org/wiki/International_Standard_Atmosphere
# See picture at https://en.wikipedia.org/wiki/NRLMSISE-00 lmao
R0 = 6356766  # m, radius of earth for geopotential calculations
g0 = 9.80665  # m/s
Rgas = 8.31432e3  # J/(kmol*K), gas constant
M0 = 28.9644  # kg/kmol, molecular weight
gamma = 1.4  # ratio of specific heats
atmo_levels = np.array(
    (-2000.63, 0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 84852, 88744)
)  # meters
pressure_refs = np.array(
    (
        108900,
        101325,
        22632.1,
        5474.89,
        868.019,
        110.9063,
        66.9389,
        3.95642,
        0.373384,
        0.183360,
    )
)  # Pa
temp_refs = np.array(
    (292.05, 288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946, 186.946)
)  # K
temp_grads = (
    np.array((-1.95, -6.5, 0, 1, 2.8, 0, -2.8, -2, 0, (730 - 187) / (150 - 90))) * 1e-3
)  # K/m


def _earth_atmo_lookup(altitude):
    H = R0 * altitude / (R0 + altitude)  # geopotential altitude
    i = max(0, np.searchsorted(atmo_levels, H) - 1)  # what entry of *_refs to use
    return H, i


def earth_atmo_temperature(altitude):
    H, i = _earth_atmo_lookup(altitude)
    return temp_refs[i] + temp_grads[i] * (H - atmo_levels[i])


def earth_atmo_pressure(altitude):
    H, i = _earth_atmo_lookup(altitude)
    if (L := temp_grads[i]) != 0.0:
        base = temp_refs[i] / (temp_refs[i] + temp_grads[i] * (H - atmo_levels[i]))
        exponent = g0 * M0 / (Rgas * L)
        return pressure_refs[i] * np.pow(base, exponent)
    else:
        return pressure_refs[i] * np.exp(
            -g0 * M0 * (H - atmo_levels[i]) / (Rgas * temp_refs[i])
        )


def earth_atmo_density(altitude):
    pressure = earth_atmo_pressure(altitude)
    temperature = earth_atmo_temperature(altitude)
    return pressure * M0 / (Rgas * temperature)


def earth_sound_speed(altitude):
    """Uses ideal gas law approximation with uniform properties."""
    return np.sqrt(gamma * Rgas * earth_atmo_temperature(altitude) / M0)


class Earth(Body):
    """Body representing the Earth including atmospheric convenience functions."""

    def __init__(self, body_state):
        super().__init__('Earth', body_state, 6371e3, actor.GM_Earth)

    def get_static_temperature(self, ship_state):
        return earth_atmo_temperature(self.get_altitude(ship_state))

    def get_static_pressure(self, ship_state):
        return earth_atmo_pressure(self.get_altitude(ship_state))

    def get_atmo_density(self, ship_state):
        return earth_atmo_density(self.get_altitude(ship_state))

    def get_mach_number(self, ship_state):
        sound_speed = earth_sound_speed(self.get_altitude(ship_state))
        return np.linalg.norm(self.get_airspeed_vector(ship_state)) / sound_speed

    def get_total_temperature(self, ship_state):
        mach = self.get_mach_number(ship_state)
        ratio = 1 + (gamma - 1) / 2 * mach**2
        return ratio * self.get_static_temperature(ship_state)

    def get_total_pressure(self, ship_state):
        mach = self.get_mach_number(ship_state)
        ratio = 1 + (gamma - 1) / 2 * mach**2
        return ratio ** (gamma / (gamma - 1)) * self.get_static_pressure(ship_state)

    def get_normal_shock_temperature(self, ship_state):
        mach = self.get_mach_number(ship_state)
        ratio = (
            (1 + (gamma - 1) / 2 * mach**2)
            * (2 * gamma / (gamma - 1) * mach**2 - 1)
            / (mach**2 * (2 * gamma / (gamma - 1) + (gamma - 1) / 2))
        )
        return ratio * self.get_static_temperature(ship_state)


class Environment:
    """Context to track simulation parameters such as colliding with celestial bodies"""

    def __init__(self, bodies: list[Body]):
        self.bodies = bodies

    def get_gravity_actors(self):
        return [body.gravity for body in self.bodies]

    def check_for_body_collisions(self, ship_state):
        """If the ship state has overlapped a body, return the body, else None"""
        for body in self.bodies:
            if body.check_collision(ship_state):
                return body
        return None


def make_environment_ECI():
    """Contains Earth centered at the origin.
    Frame is oriented with Earth's north pole along +z, similar to J2000/EME2000/ICRF frames.
    """
    return Environment([Earth(sd.make_state(angvel=(0, 0, 2 * np.pi / 86164.0905)))])


if __name__ == '__main__':
    env = make_environment_ECI()
    earth = env.bodies[0]
    ship_state = sd.state_from_orbit_properties(
        actor.GM_Earth, period=1e5, eccentricity=0, true_anomaly=-0.1
    )
    print(ship_state)
    print(earth.get_euler_angles_NED(ship_state))  # 90deg, not quite 90deg, 90deg

    import matplotlib.pyplot as plt

    alts = np.linspace(-5e3, 140e3)
    temp = tuple(earth_atmo_temperature(alt) for alt in alts)
    pres = tuple(earth_atmo_pressure(alt) for alt in alts)
    dens = tuple(earth_atmo_density(alt) for alt in alts)
    fig, axs = plt.subplots(ncols=3, sharey=True)
    axt, axp, axr = axs
    axt.plot(temp, alts)
    axp.semilogx(pres, alts)
    axr.semilogx(dens, alts)
    # axp.plot(pres, alts)
    # axr.plot(dens, alts)
    # axt.set_ylim(0, 100e3)
    axt.set_ylabel('Altitude from sea level (m)')
    axt.set_xlabel('Temp (K)')
    axp.set_xlabel('Pressure (Pa)')
    axr.set_xlabel('Density (kg/m^3)')
    for ax in axs:
        ax.grid()
    fig.tight_layout()
    plt.show()
