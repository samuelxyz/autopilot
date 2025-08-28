
import numpy as np
import quaternion as quat
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.ticker as tck

import ship, thruster, controller, sensor, sim_tools, environment, actor, graph_tools
import state_def as sd

def scenario_ideal_binary_control():
    # thruster_pack = thruster.Thrusters_Floating_6()
    # controller_ = controller.PID(np.zeros(3), (0.5, 0, 5), (3,), thruster_pack)
    thruster_accel = 2
    thruster_pack = thruster.Thrusters_Fixed_Binary_6(thruster_accel)
    controller_ = controller.Binary_Controller_3(sd.make_state(), thruster_accel, thruster_pack)
    state = sd.make_state()
    state[sd.VEL] = 2.*np.ones(3)
    ship_ = ship.Ship(state, 1, np.eye(3), controller_, actors=[thruster_pack,])

    num_steps = 100
    time = np.linspace(0, 10, num_steps)
    values_to_track = {
        # name: (reference, optional shape to be used if reference is uninitialized)
        'true_state': ship_.state,
        'thrusts': lambda: thruster_pack.thrusts, # lambda is needed because apparently np.zeros() produces something that gets stored as a value instead of a reference? or something. anyway it doesnt update properly without the lambda
    }
    hist = sim_tools.History(num_steps, values_to_track)
    sim_tools.run_simulation(time, ship_.update, hist)
    
    fig, ax = plt.subplots()
    ax.plot(time, hist['true_state'][:, sd.POS.start], label='pos.x (m)')    
    ax.plot(time, hist['true_state'][:, sd.VEL.start], label='vel.x (m/s)')
    ax.plot(time, hist['thrusts'][:, 0], label='thruster.x (m/s^2)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid()
    plt.show()

def scenario_sensing():
    randseed = 1234
    num_steps = 200
    sim_duration = 20
    time = np.linspace(0, sim_duration, num_steps)

    env_ECI = environment.make_environment_ECI()

    gyro = sensor.Gyroscope(1e-3, randseed)
    star_tracker = sensor.StarTracker(1e-4, 1, randseed)
    # Not quite the same orbits as real GPS
    gps_n_per_plane = 2
    gps_planes = 3
    gps_sats = [sensor.DistanceSignal(
        sd.state_from_orbit_properties(
            actor.GM_Earth, period=86400/2, eccentricity=0, inclination=1, 
            long_asc_node=2*np.pi*p/gps_planes, true_anomaly=2*np.pi*i/gps_n_per_plane
        ),
        env_ECI.get_gravity_actors(), 0.1, 2, 1234
        ) for i in range(gps_n_per_plane) for p in range(gps_planes)
    ]
    sds = sensor.SensorSystem_EKF(
        [gyro, star_tracker] + gps_sats, 
        sd.make_state(pos=(1e7, 0, 0)), # initial estimate. Note the filter can diverge if too far off
        np.diag([1e9, 1e9, 1e9, 1, 1, 1, 1, 1e6, 1e6, 1e6, 1, 1, 1]), # initial P - uninformative
        1e-7 * np.eye(sd.STATE_N) # if Q is zero then P tends to become singular in the Kalman filter
    )

    ship_state = sd.state_from_orbit_properties(actor.GM_Earth, period=10_000, eccentricity=0,
                                                body_angvel=(0.1, 0.1, 0.1))
    ship_ = ship.Ship(ship_state, 1, np.eye(3), state_det_system=sds, actors=env_ECI.get_gravity_actors())

    hist = sim_tools.History(num_steps, {
        'true_state': lambda: ship_.state,
        'est_state': lambda: sds.get_current_state_estimate()[0],
        'est_state_cov': lambda: sds.get_current_state_estimate()[1],
        # 'est_state': lambda: sds.EKF.x,
        # 'est_state_cov': lambda: sds.EKF.P,
    })
    def update_simulation(step_start_time, dt):
        ship_.update(step_start_time, dt)
        for sat in gps_sats:
            sat.update_simulation(dt)
        if b:= env_ECI.check_for_body_collisions(ship_.state) is not None:
            hist.add_flag(sim_tools.Flag(
                f'Ship collided with {b}', step_start_time + dt, b, end_sim=True))
    sim_tools.run_simulation(time, update_simulation, hist)
    grapher = graph_tools.Grapher(time, hist['true_state'], hist['est_state'], hist['est_state_cov'])
    
    fig, axs = plt.subplots(2, 2, sharex=True)
    (ax_pos, ax_poserr), (ax_ang, ax_ori) = axs

    ax_pos.plot(time, hist['true_state'][:, sd.POS.start], label='True pos_x')
    grapher.plot_est_component(ax_pos, sd.POS.start,'Est pos_x', color='C1')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.legend()

    grapher.plot_component_errs(ax_poserr, sd.POS, 'error_pos', list('xyz'))
    ax_poserr.set_ylabel('Position Error (m)')

    ax_ang.plot(time, hist['true_state'][:, sd.ANGVEL.start], label='True omega_x')
    grapher.plot_est_component(ax_ang, sd.ANGVEL.start,'Est omega_x', 'C1')
    ax_ang.set_ylabel('Angular velocity\n(rad/s)')
    ax_ang.legend()
    ax_ang.set_ylim(0.09, 0.11)

    grapher.plot_orientation_err(ax_ori, 'C1')
    ax_ori.set_ylim(0, 0.005)
    ax_ori.yaxis.set_major_formatter(tck.EngFormatter('rad'))
    ax_ori.legend()
    fig.tight_layout()

    plt.show()

def scenario_EDL():
    num_steps = 2000
    sim_duration = 1000
    time = np.linspace(0, sim_duration, num_steps)

    env_ECI = environment.make_environment_ECI()
    earth: environment.Earth = env_ECI.bodies[0]
    ship_state = sd.state_from_orbit_properties(
        actor.GM_Earth, periapsis=6441e3, apoapsis=6441e3, arg_periapsis=-np.pi/2)
    cs_area = 20 # m^2
    base_drag = 0.3
    drag_eq = actor.Drag_Equation(base_drag, cs_area, earth.get_atmo_density, earth.get_airspeed_vector)
    ship_ = ship.Ship(ship_state, 10e3, 40e3*np.eye(3), actors=env_ECI.get_gravity_actors() + [drag_eq])
    heat_shield_mcp = 4e3 # J/K
    heat_shield_area = cs_area
    heat_shield_temp = 300 # K
    convective_flux = np.nan
    radiative_flux = np.nan
    hist = sim_tools.History(num_steps, {
        'true_state': lambda: ship_.state,
        'g_load': lambda: np.linalg.norm(ship_.calc_apparent_accel(ship_.state)[sd.LIN])/9.80665,
        'altitude': lambda: earth.get_altitude(ship_.state),
        'mach': lambda: earth.get_mach_number(ship_.state),
        'shock_temp': lambda: earth.get_normal_shock_temperature(ship_.state),
        'stagnation_temp': lambda: earth.get_total_temperature(ship_.state),
        'convective_flux': lambda: convective_flux,
        'radiative_flux': lambda: radiative_flux,
        'heat_shield_temp': lambda: heat_shield_temp,
        'density': lambda: earth.get_atmo_density(ship_.state),
    })
    chutes_deployed = False
    chutes_opened_time = None
    # chute_easing_function = lambda x: np.sin(x*np.pi/2)**2 if x < 1 else 1
    chute_easing_function = lambda x: x**3 if x < 1 else 1
    def update_simulation(step_start_time, dt):
        nonlocal heat_shield_temp
        nonlocal convective_flux
        nonlocal radiative_flux

        ship_.update(step_start_time, dt)
        time_now = step_start_time + dt

        density = earth.get_atmo_density(ship_.state)
        heat_transfer_coeff = 3*np.sqrt(density) # wild approximation + guess, W/(m^2 K)
        env_temp = earth.get_total_temperature(ship_.state)
        temp_diff = env_temp - heat_shield_temp
        convective_flux = heat_transfer_coeff * temp_diff # W/m^2
        radiative_flux = 0 # one day perhaps i will put some kind of wild clever bodge approximation in here
        heat_shield_temp += (convective_flux + radiative_flux) * heat_shield_area * dt / heat_shield_mcp

        nonlocal chutes_deployed
        nonlocal chutes_opened_time
        chute_opening_pd = 30

        if earth.get_altitude(ship_.state) < 3e3 and not chutes_deployed:
            hist.add_flag(sim_tools.Flag(f'Parachute deploy started, will take {chute_opening_pd}s', time_now))
            chutes_opened_time = time_now
            chutes_deployed = True
        if chutes_deployed:
            easing_function_x = (time_now - chutes_opened_time)/chute_opening_pd
            eased_proportion = chute_easing_function(easing_function_x)
            drag_eq.cs_area = cs_area * (1 + eased_proportion*100)
            drag_eq.drag_coeff = base_drag + 0.5*eased_proportion

        if (b:= env_ECI.check_for_body_collisions(ship_.state)) is not None:
            hist.add_flag(sim_tools.Flag(
                f'Ship collided with {b.name} at {np.linalg.norm(b.get_airspeed_vector(ship_.state)):.1f} m/s', time_now, b, end_sim=True))
    sim_tools.run_simulation(time, update_simulation, hist)
    hist.print_flags()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    (ax_prof, ax_altmach), (ax_alttemp, ax_alttime) = axs
    for axrow in axs:
        for ax in axrow:
            ax.grid(True)
    markevery = 100
    ax_prof.plot(hist['true_state'][:, sd.POS.start], hist['true_state'][:, sd.POS.start+1], 'x',
                 linestyle='-', label='Trajectory', markevery=60, color='tab:red')
    ax_prof.set_xlabel('Position x')
    ax_prof.set_ylabel('Position y')
    ax_prof.set_aspect('equal')
    earth_ellipse = patch.Ellipse((0, 0,), 2*6371e3, 2*6371e3, color='tab:blue')
    earth_ellipse.set_clip_box(ax_prof.bbox)
    ax_prof.add_artist(earth_ellipse)
    km_formatter = tck.FuncFormatter(lambda x, pos: f'{int(x/1000)} km')
    ax_prof.xaxis.set_major_formatter(km_formatter)
    ax_prof.yaxis.set_major_formatter(km_formatter)
    ax_prof.tick_params("x", rotation=90)
    ax_prof.tick_params("y", rotation=45)
    ax_prof.set_title(f'Trajectory - Marks every {markevery*sim_duration/num_steps}s')

    ax_altmach.plot(hist['mach'], hist['altitude'])
    ax_altmach.tick_params(axis='x', labelcolor='C0')
    ax_altmach.set_xlabel('Mach number', color='C0')
    ax_altmach.set_ylabel('Altitude')
    ax_altmach.yaxis.set_major_formatter(tck.EngFormatter('m'))
    ax_altmach.set_xlim(0)
    ax_altmach.set_ylim(0)

    gee_color='C1'
    ax_altgee = ax_altmach.twiny()
    ax_altgee.plot(hist['g_load'], hist['altitude'], color=gee_color)
    ax_altgee.tick_params(axis='x', labelcolor=gee_color)
    ax_altgee.set_xlabel('G-force', color=gee_color)
    ax_altgee.set_ylabel('Altitude')
    ax_altgee.set_xlim(0)

    heat_color='C2'
    ax_altheat = ax_alttemp.twiny()
    ax_altheat.plot(hist['convective_flux'], hist['altitude'], color=heat_color, label='Convective')
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

    fig.tight_layout()
    plt.show()
if __name__ == '__main__':
    # scenario_ideal_binary_control()
    # scenario_sensing()
    scenario_EDL()