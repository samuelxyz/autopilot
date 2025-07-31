
import itertools
import numpy as np
import quaternion as quat
import matplotlib.pyplot as plt

import ship, thruster, controller, sensor
import state_def as sd
import sim_tools

def scenario_ideal_binary_control():
    # thruster_pack = thruster.Thrusters_Floating_6()
    # controller_ = controller.PID(np.zeros(3), (0.5, 0, 5), (3,), thruster_pack)
    thruster_accel = 2
    thruster_pack = thruster.Thrusters_Fixed_Binary_6(thruster_accel)
    controller_ = controller.Binary_Controller_3(sd.make_zero_state(), thruster_accel, thruster_pack)
    state = sd.make_zero_state()
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

    state = sd.make_zero_state()
    state[sd.ANGVEL] = [0.1, 0.1, 0.1]
    gyro = sensor.Gyroscope(0.01, randseed)
    star_tracker = sensor.StarTracker(1e-6, 1, randseed)
    sds = sensor.StateDeterminationSystem(
        [gyro, star_tracker], sd.make_zero_state(), 0.1*np.eye(sd.STATE_N), 
        1e-4 * np.eye(sd.STATE_N) # if Q is zero then P tends to become singular in the Kalman filter
    )
    ship_ = ship.Ship(state, 1, np.eye(3), state_det_system=sds)

    hist = sim_tools.History(num_steps, {
        'true_state': ship_.state,
        'est_state': lambda: sds.get_current_state_estimate()[0],
        'est_state_cov': lambda: sds.get_current_state_estimate()[1],
    })
    sim_tools.run_simulation(time, ship_.update, hist)

    ang_est = hist['est_state'][:, sd.ANGVEL.start]
    ang_sd = np.sqrt(hist['est_state_cov'][:, sd.ANGVEL.start, sd.ANGVEL.start])
    ori_true = sd.get_orient(hist['true_state'])
    ori_est = sd.get_orient(hist['est_state'])
    ori_err = 2*np.acos(quat.as_float_array(ori_true * ori_est.conj())[:, 0]) # radians from est to true
    ori_component_sd = np.sqrt(np.diagonal(hist['est_state_cov'][:, sd.ORIENT, sd.ORIENT], axis1=-2, axis2=-1))
    ori_rss_z = np.linalg.norm(quat.as_float_array(ori_true - ori_est)/ori_component_sd, axis=-1)
    ori_stdev = ori_err/ori_rss_z
    
    fig, axs = plt.subplots(2, sharex=True)
    ax_ang, ax_ori = axs
    ax_ang.plot(time, hist['true_state'][:, sd.ANGVEL.start], label='True omega_x')
    ax_ang.plot(time, ang_est, label='Est omega_x')
    ax_ang.fill_between(time, ang_est - ang_sd, ang_est + ang_sd, alpha=0.2, color='C1')
    ax_ang.set_ylabel('Angular velocity (rad/s^2)')
    ax_ang.legend()
    ax_ang.grid()
    ax_ang.set_ylim(0, 0.2)
    ax_ori.plot(time, ori_err, label='True Orientation Error')
    ax_ori.plot(time, ori_stdev, label='Est Orientation Uncertainty')
    ax_ori.grid()
    ax_ori.set_xlabel('Time (s)')
    ax_ori.set_ylabel('Orientation error (rad)')
    ax_ori.set_ylim(0, 0.2)
    ax_ori.legend()
    # ori2_color='C1'
    # ax_or2 = ax_ori.twinx()
    # ax_or2.semilogy(time, ori_rss_z, color=ori2_color)
    # ax_or2.set_ylabel('Orientation error z-score\n(Components RMS)', color=ori2_color)
    # ax_or2.tick_params(axis='y', labelcolor=ori2_color)
    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    # scenario_ideal_binary_control()
    scenario_sensing()