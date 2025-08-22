from matplotlib.axes import Axes
import numpy as np
import quaternion as quat

import state_def as sd

class Grapher:

    def __init__(self, time, true_state, est_state, est_state_cov):
        '''Set up some values to use in a graphing session.
        
        Parameters:
            time: numpy array, N, to use as x-axis
            true_state: numpy array, N x STATE_N
            est_state: numpy array, N x STATE_N
            state_cov: numpy array, N x STATE_N x STATE_N
        '''
        self.uncertainty_transparency = 0.2
        self.time_label = 'Time (s)'
        self.time = time
        self.true_state = true_state
        self.est_state = est_state
        self.est_state_cov = est_state_cov

    def plot_est_component(self, ax: Axes, component_index, label, color=None):
        '''Plot the specified components of an estimated state vector on the given axes, 
        including uncertainty from a covariance matrix. Sets grid.
        
        Parameters:
            ax: matplotlib axes object on which to plot
            component_index: index into each state vector of the component to plot.
            label: legend label
            color: plot color to use
        '''
        line = self.est_state[:, component_index]
        stdev = np.sqrt(self.est_state_cov[:, component_index, component_index])
        ax.plot(self.time, line, label=label, color=color)
        ax.fill_between(self.time, line - stdev, line + stdev, 
                            alpha=self.uncertainty_transparency, color=color)
        ax.set_xlabel(self.time_label)
        ax.grid(True)

    def plot_component_errs(self, ax: Axes, component_index, vec_label, component_names=None):
        '''Plot a summary of vector error on the given axes. Sets grid, legend, and x-axis title.
        
        Parameters:
            ax: matplotlib axes object on which to plot
            component_index: length-3 slice object into state vectors
            vec_label: string with which to label the vector
            component_names: list[string] with which to label each component. defaults to ["0", "1", "2", ...]
        '''
        err_components = self.est_state[:, component_index] - self.true_state[:, component_index]
        std_components = np.sqrt(np.diagonal(self.est_state_cov[:, component_index, component_index], axis1=-2, axis2=-1))
        range_low = err_components - std_components
        range_high = err_components + std_components

        num_components = np.shape(err_components)[-1]
        if component_names is None: 
            component_names = [str(i) for i in range(num_components)]
        assert len(component_names) == num_components

        for i, name  in enumerate(component_names):
            ax.plot(self.time, err_components[:, i], label=f'{vec_label}_{name}')
            ax.fill_between(self.time, 0, range_low[:, i], range_high[:, i], alpha=self.uncertainty_transparency)
        
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(self.time_label)

    def plot_orientation_err(self, ax: Axes, color=None):
        '''Plot orientation error on the given axes. Sets grid, legend, and axis titles.
        
        Parameters:
            ax: matplotlib axes object on which to plot
        '''
        ori_true = sd.get_orient(self.true_state)
        ori_est = sd.get_orient(self.est_state)
        ori_err_quat = quat.as_float_array(ori_true * ori_est.conj())
        ori_err_angle = 2*np.acos(ori_err_quat[:, 0]) # radians from est to true

        ax.plot(self.time, ori_err_angle, label='True Orientation Error', color=color)

        if self.est_state_cov is not None:
            ori_component_sd = np.sqrt(
                np.diagonal(self.est_state_cov[:, sd.ORIENT, sd.ORIENT], axis1=-2, axis2=-1))
            ori_rss_z = np.linalg.norm(quat.as_float_array(ori_true - ori_est)/ori_component_sd, axis=-1)
            ori_stdev = ori_err_angle/ori_rss_z
            ax.fill_between(self.time, 0, ori_stdev, label='Est Orientation Uncertainty', 
                            color=color, alpha=self.uncertainty_transparency)
        ax.grid(True)
        ax.set_xlabel(self.time_label)
        ax.set_ylabel('Angle (rad)')