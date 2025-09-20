import matplotlib.pyplot as plt
import numpy as np

from autopilot import (
    controller,
    ship,
    sim_tools,
    thruster,
)
from autopilot import state_def as sd


def main():
    # thruster_pack = thruster.Thrusters_Floating_6()
    # controller_ = controller.PID(np.zeros(3), (0.5, 0, 5), (3,), thruster_pack)
    thruster_spec = 2
    thruster_pack = thruster.Thrusters_Fixed_Binary_6(thruster_spec)
    controller_ = controller.Binary_Controller_3(
        sd.make_state(), thruster_spec, thruster_pack
    )
    state = sd.make_state(vel=2.0 * np.ones(3))
    ship_ = ship.Ship(
        state,
        1,
        np.eye(3),
        controller_,
        actors=[
            thruster_pack,
        ],
    )

    num_steps = 100
    time = np.linspace(0, 10, num_steps)
    values_to_track = {
        # name: (reference, optional shape to be used if reference is uninitialized)
        'true_state': lambda: ship_.state,
        'thrusts': lambda: thruster_pack.thrusts,  # lambda is needed because apparently np.zeros() produces something that gets stored as a value instead of a reference? or something. anyway it doesnt update properly without the lambda
    }
    hist = sim_tools.History(num_steps, values_to_track)
    sim_tools.run_simulation(time, ship_.update, hist)

    fig, ax = plt.subplots()
    ax.plot(time, hist['true_state'][:, sd.POS.start], label='pos.x (m)')
    ax.plot(time, hist['true_state'][:, sd.VEL.start], label='vel.x (m/s)')
    ax.plot(time, hist['thrusts'][:, 0], label='thruster.x (N)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == '__main__':
    main()
