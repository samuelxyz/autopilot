import numpy as np
import itertools

class History(dict):
    '''Dictionary tracking the history of specified arbitrary floats or numpy arrays. 
    self[name] returns a numpy array with num_steps added as the first dimension.'''
    def __init__(self, num_steps, values_to_track: dict):
        super().__init__()
        self.num_steps = num_steps
        self.refs = {}
        self.history = {}
        for name, info in values_to_track.items():
            if isinstance(info, tuple):
                ref, shape = info
            else:
                ref = info
                value = ref() if callable(ref) else ref
                try:
                    shape = np.shape(value) # works for non-numpy numbers too
                except:
                    print(f'Could not identify shape to use for {ref} when creating history tracker - specify it in values_to_track as "name": (item, shape)')
            self.refs[name] = (ref, callable(ref))
            self[name] = np.zeros((num_steps,) + shape)
    
    def save_timestep(self, i):
        '''Access each of the tracked items, and write their current values into timestep i.'''
        for name, tracker_array in self.items():
            thing_to_be_written, should_be_called = self.refs[name]
            try:
                if should_be_called:
                    tracker_array[i] = thing_to_be_written()
                else:
                    tracker_array[i] = thing_to_be_written
            except:
                tracker_array[i] = np.nan


def run_simulation(time_arr, update_func, hist=None):
    '''Run a simulation across the given timestamps.

    Parameters:
        time_arr: Sequence of timestamps (in seconds, for example)
        update_func: Function matching the signature update_func(start_time, dt) to update simulation at each timestep, with argument dt being the length of the timestep
        hist: Optional History object to be updated after each timestep

    '''
    if hist is not None:
        hist.save_timestep(0)
    for i, (t_old, t_new) in enumerate(itertools.pairwise(time_arr)):
        dt = t_new - t_old
        update_func(t_old, dt)
        if hist is not None:
            hist.save_timestep(i+1)

class Flag:
    '''System to denote when some special event occurs in the simulation'''
    def __init__(self, text, time, end_sim=False):
        self.text=text
        self.time=time
        self.end_sim = end_sim

