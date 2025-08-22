
import numpy as np
import quaternion as quat
import matplotlib.pyplot as plt
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
    num_steps = 100
    sim_duration = 10
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
        sd.make_state(), # initial x estimate
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
    sim_tools.run_simulation(time, update_simulation, hist)
    grapher = graph_tools.Grapher(time, hist['true_state'], hist['est_state'], hist['est_state_cov'])
    
    fig, axs = plt.subplots(2, 2, sharex=True)
    (ax_pos, ax_poserr), (ax_ang, ax_ori) = axs

    ax_pos.plot(time, hist['true_state'][:, sd.POS.start], label='True pos_x')
    grapher.plot_est_component(ax_pos, sd.POS.start,'Est pos_x')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.legend()

    grapher.plot_component_errs(ax_poserr, sd.POS, 'error_pos', list('xyz'))
    ax_poserr.set_ylabel('Error (m)')

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

if __name__ == '__main__':
    # scenario_ideal_binary_control()
    scenario_sensing()