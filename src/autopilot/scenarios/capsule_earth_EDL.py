import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import quaternion as quat

from autopilot import actor, environment, graph_tools, ship, sim_tools
from autopilot import state_def as sd


def main():
    num_steps = 2000
    sim_duration = 1000
    time = np.linspace(0, sim_duration, num_steps)

    env_ECI = environment.make_environment_ECI()
    earth: environment.Earth = env_ECI.bodies[0]
    true_anomaly = -np.pi / 2
    initial_longitude = -np.pi / 2
    # arg_periapsis + true_anomaly == initial_longitude
    ship_state = sd.state_from_orbit_properties(
        actor.GM_Earth,
        periapsis=6426e3,
        apoapsis=6456e3,
        arg_periapsis=initial_longitude - true_anomaly,
        true_anomaly=true_anomaly,
    )
    NED_vel = earth.get_airspeed_vector_NED(ship_state)
    vel_heading = np.atan2(NED_vel[1], NED_vel[0])
    orient_q = earth.make_quat_from_euler_angles_NED(
        (vel_heading, 0.25, 0), ship_state[sd.POS]
    )
    ship_state[sd.ORIENT] = quat.as_float_array(orient_q)
    r_vec = ship_state[sd.POS] - earth.state[sd.POS]
    ship_state[sd.ANGVEL] = np.cross(r_vec, ship_state[sd.VEL]) / np.dot(r_vec, r_vec)
    cs_area = 20  # m^2
    capsule_drag_coeff = 0.25
    drag_eq = actor.Drag_Equation(
        capsule_drag_coeff, cs_area, earth.get_atmo_density, earth.get_airspeed_vector
    )
    lift_eq = actor.Lift_Equation(
        capsule_drag_coeff * 0.2,
        cs_area,
        earth.get_atmo_density,
        earth.get_airspeed_vector,
    )
    ship_mass = 10e3
    ship_inertia = 40e3 * np.eye(3)
    ship_ = ship.Ship(
        ship_state,
        ship_mass,
        ship_inertia,
        actors=env_ECI.get_gravity_actors() + [drag_eq, lift_eq],
    )
    heat_shield_mcp = 4e3  # J/K
    heat_shield_area = cs_area
    heat_shield_temp = 300  # K
    convective_flux = np.nan
    radiative_flux = np.nan
    hist = sim_tools.History(
        num_steps,
        {
            'true_state': lambda: ship_.state,
            'g_load': lambda: np.linalg.norm(
                ship_.calc_apparent_wrench(ship_.state)[sd.LIN]
            )
            / ship_.mass
            / 9.80665,
            'altitude': lambda: earth.get_altitude(ship_.state),
            'mach': lambda: earth.get_mach_number(ship_.state),
            'shock_temp': lambda: earth.get_normal_shock_temperature(ship_.state),
            'stagnation_temp': lambda: earth.get_total_temperature(ship_.state),
            'convective_flux': lambda: convective_flux,
            'radiative_flux': lambda: radiative_flux,
            'heat_shield_temp': lambda: heat_shield_temp,
            'density': lambda: earth.get_atmo_density(ship_.state),
            'lift': lambda: lift_eq.get_wrench(
                ship_.state, ship_.mass, ship_.inertia_matrix
            ),
            'drag': lambda: drag_eq.get_wrench(
                ship_.state, ship_.mass, ship_.inertia_matrix
            ),
            'HEB': lambda: earth.get_euler_angles_NED(ship_.state),
            'angle_of_attack': lambda: earth.get_angle_of_attack(ship_.state),
        },
    )

    def aoa_function(a):
        # crazy feel-based definition
        # Maximum is 0.5exp(-0.5) ~ 0.303 at aoa=0.5
        return a * np.exp(-2 * a**2)

    chutes_deployed = False
    chutes_opened_time = None
    chute_trigger_altitude = 4e3
    chute_opening_pd = 30

    # chute_easing_function = lambda x: np.sin(x*np.pi/2)**2 if x < 1 else 1
    def chute_easing_function(x):
        return x**3 if x < 1 else 1

    def update_simulation(step_start_time, dt):
        nonlocal heat_shield_temp
        nonlocal convective_flux
        nonlocal radiative_flux

        ship_.update(step_start_time, dt)
        time_now = step_start_time + dt

        aoa = earth.get_angle_of_attack(ship_.state)
        lift_eq.lift_coeff = aoa_function(aoa) * capsule_drag_coeff

        density = earth.get_atmo_density(ship_.state)
        heat_transfer_coeff = 3 * np.sqrt(
            density
        )  # wild approximation + guess, W/(m^2 K)
        env_temp = earth.get_total_temperature(ship_.state)
        temp_diff = env_temp - heat_shield_temp
        convective_flux = heat_transfer_coeff * temp_diff  # W/m^2
        radiative_flux = 0  # one day perhaps i will put some kind of wild clever bodge approximation in here
        heat_shield_temp += (
            (convective_flux + radiative_flux) * heat_shield_area * dt / heat_shield_mcp
        )

        nonlocal chutes_deployed
        nonlocal chutes_opened_time

        if (
            earth.get_altitude(ship_.state) < chute_trigger_altitude
        ) and not chutes_deployed:
            hist.add_flag(
                sim_tools.Flag(
                    f'Parachute deploy started, will take {chute_opening_pd}s', time_now
                )
            )
            chutes_opened_time = time_now
            chutes_deployed = True
        if chutes_deployed:
            easing_function_x = (time_now - chutes_opened_time) / chute_opening_pd
            eased_proportion = chute_easing_function(easing_function_x)
            drag_eq.cs_area = cs_area * (1 + eased_proportion * 100)
            drag_eq.drag_coeff = capsule_drag_coeff + 0.5 * eased_proportion

        if (b := env_ECI.check_for_body_collisions(ship_.state)) is not None:
            hist.add_flag(
                sim_tools.Flag(
                    f'Ship collided with {b.name} at {np.linalg.norm(b.get_airspeed_vector(ship_.state)):.1f} m/s',
                    time_now,
                    b,
                    end_sim=True,
                )
            )

    sim_tools.run_simulation(time, update_simulation, hist)
    hist.print_flags()

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    (ax_traj_xy, ax_alttime, ax_altmach), (ax_traj_xz, ax_orient, ax_alttemp) = axs
    for axrow in axs:
        for ax in axrow:
            ax.grid(True)

    markevery = 100
    ax_traj_xy.plot(
        hist['true_state'][:, sd.POS.start],
        hist['true_state'][:, sd.POS.start + 1],
        'x',
        linestyle='-',
        label='Trajectory',
        markevery=60,
        color='tab:red',
    )
    ax_traj_xy.set_xlabel('Position x')
    ax_traj_xy.set_ylabel('Position y')
    ax_traj_xy.set_aspect('equal')
    earth_ellipse_xy = patch.Ellipse(
        (
            0,
            0,
        ),
        2 * 6371e3,
        2 * 6371e3,
        color='tab:blue',
    )
    earth_ellipse_xy.set_clip_box(ax_traj_xy.bbox)
    ax_traj_xy.add_artist(earth_ellipse_xy)
    km_formatter = tck.FuncFormatter(lambda x, pos: f'{int(x / 1000)} km')
    ax_traj_xy.xaxis.set_major_formatter(km_formatter)
    ax_traj_xy.yaxis.set_major_formatter(km_formatter)
    ax_traj_xy.tick_params('x', rotation=90)
    ax_traj_xy.tick_params('y', rotation=45)
    ax_traj_xy.set_title(
        f'Trajectory - Marks every {markevery * sim_duration / num_steps}s'
    )

    ax_traj_xz.plot(
        hist['true_state'][:, sd.POS.start],
        hist['true_state'][:, sd.POS.start + 2],
        'x',
        linestyle='-',
        label='Trajectory',
        markevery=60,
        color='tab:red',
    )
    ax_traj_xz.set_xlabel('Position x')
    ax_traj_xz.set_ylabel('Position z')
    ax_traj_xz.set_ylim(-1000e3, 1000e3)
    ax_traj_xz.set_aspect('equal')
    earth_ellipse_xz = patch.Ellipse(
        (
            0,
            0,
        ),
        2 * 6371e3,
        2 * 6371e3,
        color='tab:blue',
    )
    earth_ellipse_xz.set_clip_box(ax_traj_xz.bbox)
    ax_traj_xz.add_artist(earth_ellipse_xz)
    ax_traj_xz.xaxis.set_major_formatter(km_formatter)
    ax_traj_xz.yaxis.set_major_formatter(km_formatter)
    ax_traj_xz.tick_params('x', rotation=90)
    ax_traj_xz.tick_params('y', rotation=45)

    ax_altmach.plot(hist['mach'], hist['altitude'])
    ax_altmach.tick_params(axis='x', labelcolor='C0')
    ax_altmach.set_xlabel('Mach number', color='C0')
    ax_altmach.set_ylabel('Altitude')
    ax_altmach.yaxis.set_major_formatter(tck.EngFormatter('m'))
    ax_altmach.set_xlim(0)
    ax_altmach.set_ylim(0)

    gee_color = 'C1'
    ax_altgee = ax_altmach.twiny()
    ax_altgee.plot(hist['g_load'], hist['altitude'], color=gee_color)
    ax_altgee.tick_params(axis='x', labelcolor=gee_color)
    ax_altgee.set_xlabel('G-force', color=gee_color)
    ax_altgee.set_ylabel('Altitude')
    ax_altgee.set_xlim(0)

    heat_color = 'C2'
    ax_altheat = ax_alttemp.twiny()
    ax_altheat.plot(
        hist['convective_flux'], hist['altitude'], color=heat_color, label='Convective'
    )
    # ax_altheat.plot(hist['radiative_flux'], hist['altitude'], color=heat_color, linestyle=':', label='Radiative')
    ax_altheat.tick_params(axis='x', labelcolor=heat_color)
    ax_altheat.set_xlabel('Heat flux (W/m^2)', color=heat_color)
    ax_altheat.set_ylabel('Altitude')
    ax_altheat.set_ylim(0)

    temp_plotter = ax_alttemp.semilogx
    # temp_plotter(hist['shock_temp'], hist['altitude'], label='Normal Shock')
    temp_plotter(hist['stagnation_temp'], hist['altitude'], label='Stagnation Temp')
    temp_plotter(hist['heat_shield_temp'], hist['altitude'], label='Heat Shield')
    ax_alttemp.set_xlabel('Temperature')
    ax_alttemp.set_ylabel('Altitude')
    ax_alttemp.yaxis.set_major_formatter(tck.EngFormatter('m'))
    ax_alttemp.xaxis.set_major_formatter(tck.EngFormatter('K'))
    # ax_alttemp.set_xlim(0)
    ax_alttemp.set_ylim(0)
    ax_alttemp.legend()

    ax_alttime.plot(time, hist['altitude'])
    graph_tools.plot_flags(ax_alttime, hist)
    ax_alttime.set_xlabel('Elapsed Time')
    ax_alttime.set_ylabel('Altitude')
    ax_alttime.xaxis.set_major_formatter(tck.EngFormatter('s'))
    ax_alttime.yaxis.set_major_formatter(tck.EngFormatter('m'))
    ax_alttime.set_xlim(0)
    ax_alttime.set_ylim(0)

    ax_orient.plot(time, hist['HEB'][:, 0], label='Heading (0 is North)')
    ax_orient.plot(time, hist['HEB'][:, 1], label='Elevation (0 is level)')
    ax_orient.plot(time, hist['HEB'][:, 2], label='Bank (0 is level)')
    ax_orient.plot(time, hist['angle_of_attack'], label='Angle of Attack')
    ax_orient.set_xlabel('Elapsed Time')
    ax_orient.set_ylabel('Orientation Angles (rad)')
    ax_orient.set_xlim(0)
    ax_orient.legend(title='Note: Heat shield normal is "forward"')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
