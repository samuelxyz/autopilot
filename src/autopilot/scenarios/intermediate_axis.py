import matplotlib.pyplot as plt
import numpy as np

from autopilot import ship, sim_tools
from autopilot import state_def as sd


def main():
    num_steps = 1000
    sim_duration = 20
    time = np.linspace(0, sim_duration, num_steps)

    ship_ = ship.Ship(
        state=sd.make_state(angvel=[8, 1e-3, 1e-3]),
        mass=1,
        inertia_matrix=np.diag([2, 1, 2.2]),
    )

    hist = sim_tools.History(
        num_steps,
        {
            'HEB': lambda: sd.get_heading_elevation_bank_FRB(
                sd.get_orient_q(ship_.state)
            ),
            'omega_world': lambda: ship_.state[sd.ANGVEL],
            'omega_body': lambda: sd.rotate_vector(
                sd.get_orient_q(ship_.state).conj(), ship_.state[sd.ANGVEL]
            ),
        },
    )
    sim_tools.run_simulation(time, ship_.update, hist)
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    ax_orient, ax_om_w, ax_om_b = axs

    ax_orient.plot(time, hist['HEB'][:, 0], label='Yaw (xy-plane, 0 is +x)')
    ax_orient.plot(
        time, hist['HEB'][:, 1], label='Pitch (0 is in xy-plane, max is facing -z)'
    )
    ax_orient.plot(time, hist['HEB'][:, 2], label='Roll')
    ax_orient.set_xlabel('Elapsed Time')
    ax_orient.set_ylabel('Orientation Angles (rad)')

    ax_om_w.plot(time, hist['omega_world'][:, 0], label='omega_x_world')
    ax_om_w.plot(time, hist['omega_world'][:, 1], label='omega_y_world')
    ax_om_w.plot(time, hist['omega_world'][:, 2], label='omega_z_world')
    ax_om_w.set_xlabel('Elapsed Time')
    ax_om_w.set_ylabel('Angular Velocity (rad/s)')

    ax_om_b.plot(time, hist['omega_body'][:, 0], label='omega_x_body')
    ax_om_b.plot(time, hist['omega_body'][:, 1], label='omega_y_body')
    ax_om_b.plot(time, hist['omega_body'][:, 2], label='omega_z_body')
    ax_om_b.set_xlabel('Elapsed Time')
    ax_om_b.set_ylabel('Angular Velocity (rad/s)')

    for ax in axs:
        ax.grid()
        ax.legend()
        ax.set_xlim(0)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
