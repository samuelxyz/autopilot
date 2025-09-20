import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np

from autopilot import actor, environment, graph_tools, sensor, ship, sim_tools
from autopilot import state_def as sd


def main():
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
    gps_sats = [
        sensor.DistanceSignal(
            sd.state_from_orbit_properties(
                actor.GM_Earth,
                period=86400 / 2,
                eccentricity=0,
                inclination=1,
                long_asc_node=2 * np.pi * p / gps_planes,
                true_anomaly=2 * np.pi * i / gps_n_per_plane,
            ),
            env_ECI.get_gravity_actors(),
            0.1,
            2,
            1234,
        )
        for i in range(gps_n_per_plane)
        for p in range(gps_planes)
    ]
    sds = sensor.SensorSystem_EKF(
        [gyro, star_tracker] + gps_sats,
        sd.make_state(
            pos=(1e7, 0, 0)
        ),  # initial estimate. Note the filter can diverge if too far off
        np.diag(
            [1e9, 1e9, 1e9, 1, 1, 1, 1, 1e6, 1e6, 1e6, 1, 1, 1]
        ),  # initial P - uninformative
        1e-7
        * np.eye(
            sd.STATE_N
        ),  # if Q is zero then P tends to become singular in the Kalman filter
    )

    ship_state = sd.state_from_orbit_properties(
        actor.GM_Earth, period=10_000, eccentricity=0, body_angvel=(0.1, 0.1, 0.1)
    )
    ship_ = ship.Ship(
        ship_state,
        1,
        np.eye(3),
        state_det_system=sds,
        actors=env_ECI.get_gravity_actors(),
    )

    hist = sim_tools.History(
        num_steps,
        {
            'true_state': lambda: ship_.state,
            'est_state': lambda: sds.get_current_state_estimate()[0],
            'est_state_cov': lambda: sds.get_current_state_estimate()[1],
            # 'est_state': lambda: sds.EKF.x,
            # 'est_state_cov': lambda: sds.EKF.P,
        },
    )

    def update_simulation(step_start_time, dt):
        ship_.update(step_start_time, dt)
        for sat in gps_sats:
            sat.update_simulation(dt)
        if b := env_ECI.check_for_body_collisions(ship_.state) is not None:
            hist.add_flag(
                sim_tools.Flag(
                    f'Ship collided with {b}', step_start_time + dt, b, end_sim=True
                )
            )

    sim_tools.run_simulation(time, update_simulation, hist)
    grapher = graph_tools.Grapher(
        time, hist['true_state'], hist['est_state'], hist['est_state_cov']
    )

    fig, axs = plt.subplots(2, 2, sharex=True)
    (ax_pos, ax_poserr), (ax_ang, ax_ori) = axs

    ax_pos.plot(time, hist['true_state'][:, sd.POS.start], label='True pos_x')
    grapher.plot_est_component(ax_pos, sd.POS.start, 'Est pos_x', color='C1')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.legend()

    grapher.plot_component_errs(ax_poserr, sd.POS, 'error_pos', list('xyz'))
    ax_poserr.set_ylabel('Position Error (m)')

    ax_ang.plot(time, hist['true_state'][:, sd.ANGVEL.start], label='True omega_x')
    grapher.plot_est_component(ax_ang, sd.ANGVEL.start, 'Est omega_x', 'C1')
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
    main()
